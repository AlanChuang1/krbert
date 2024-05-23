import json
import pandas as pd
from torch.utils.data import Dataset
import torch
from collections import defaultdict
import math
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

def load_reviews(review_file):
    reviews = []
    with open(review_file, 'r') as file:
        for line in file:
            reviews.append(json.loads(line.strip()))
    return pd.DataFrame(reviews)

def load_metadata(meta_file):
    metadata = []
    with open(meta_file, 'r') as file:
        for line in file:
            metadata.append(json.loads(line.strip()))
    return pd.DataFrame(metadata)

def create_triplets(reviews_df, metadata_df):
    # stratified_sample = reviews_df.groupby(['category', 'rating'], group_keys=False).apply(lambda x: x.sample(min(len(x), 10000 // (len(reviews_df['category'].unique()) * 5))))
    triplets = []
    
    for _, row in reviews_df.iterrows():
        asin = row['asin']
        
        # Product-Category Relationship
        triplets.append((asin, 'categorized_as', 'All_Beauty'))
        
        # Product-Rating Relationship
        triplets.append((asin, 'rated_as', str(row['rating'])))
        
        # Product-User Relationship
        triplets.append((row['user_id'], 'reviewed', asin))
        
        # Product-Title Relationship
        if 'title' in row and row['title']:
            triplets.append((asin, 'has_title', row['title']))
        
        # Product-Text Relationship
        if 'text' in row and row['text']:
            triplets.append((asin, 'has_text', row['text']))
        
        # Product-Helpfulness Relationship
        triplets.append((asin, 'has_helpful_vote', str(row['helpful_vote'])))
        
        # Product-Verified Purchase Relationship
        triplets.append((asin, 'verified_purchase', str(row['verified_purchase'])))
        
    for _, row in metadata_df.iterrows():
        parent_asin = row['parent_asin']
        
        # Product-Price Relationship
        if 'price' in row and row['price'] is not None:
            triplets.append((parent_asin, 'priced_at', str(row['price'])))
        
        # Product-Feature Relationship
        if 'features' in row and row['features']:
            for feature in row['features']:
                triplets.append((parent_asin, 'has_feature', feature))
        
        # Product-Description Relationship
        if 'description' in row and row['description']:
            for desc in row['description']:
                triplets.append((parent_asin, 'described_as', desc))
        
        # Product-Store Relationship
        if 'store' in row and row['store']:
            triplets.append((parent_asin, 'sold_by', row['store']))
        
        # Product-Detail Relationship
        if 'details' in row and row['details']:
            for key, value in row['details'].items():
                triplets.append((parent_asin, f'has_detail_{key.lower()}', value))

    return triplets

def calculate_frequency_scores(triplets):
    entity_counts = Counter([triplet[0] for triplet in triplets])
    max_count = max(entity_counts.values())
    frequency_scores = {entity: count / max_count for entity, count in entity_counts.items()}
    return frequency_scores

def calculate_context_scores(triplets):
    # Collect all contexts for each entity
    entity_contexts = defaultdict(list)
    for entity, relation, related_entity in triplets:
        entity_contexts[entity].append((relation, related_entity))
    
    # Calculate consistency scores
    context_scores = {}
    for entity, contexts in entity_contexts.items():
        context_counts = defaultdict(int)
        for context in contexts:
            context_counts[context] += 1
        
        # Calculate entropy as a measure of consistency
        total_contexts = len(contexts)
        entropy = -sum((count / total_contexts) * math.log2(count / total_contexts) for count in context_counts.values())
        max_entropy = math.log2(total_contexts) if total_contexts > 1 else 1  # Avoid division by zero
        
        # Consistency score as 1 - normalized entropy (lower entropy means higher consistency)
        context_scores[entity] = 1 - (entropy / max_entropy)
    
    return context_scores

def calculate_source_scores(triplets):
    source_scores = {}
    for triplet in triplets:
        entity, relation, value = triplet
        if relation == 'verified_purchase':
            score = 1.0 if value == 'True' else 0.5
            source_scores[entity] = score
    return source_scores

def calculate_confidence_scores(triplets, alpha=0.33, beta=0.33, gamma=0.34):
    frequency_scores = calculate_frequency_scores(triplets)
    context_scores = calculate_context_scores(triplets)
    source_scores = calculate_source_scores(triplets)
    
    confidence_scores = {}
    for entity in frequency_scores.keys():
        confidence_score = (alpha * frequency_scores.get(entity, 0) +
                            beta * context_scores.get(entity, 0) +
                            gamma * source_scores.get(entity, 0))
        confidence_scores[entity] = confidence_score
    return confidence_scores
    

class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, confidence_scores=None, use_confidence=False, max_length=128):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.confidence_scores = confidence_scores
        self.use_confidence = use_confidence
        self.relation_vocab = {rel: idx for idx, rel in enumerate(set([t[1] for t in triplets]))}

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        entity1, relation, entity2 = self.triplets[idx]
        inputs = self.tokenizer(
            text=entity1, text_pair=entity2, add_special_tokens=True,
            max_length=self.max_length, padding='max_length', truncation=True,
            return_tensors='pt'
        )
        label = self.get_label(entity1, relation, entity2) 

        item = {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'relation': torch.tensor(self.relation_vocab[relation], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float) 
        }

        if self.use_confidence:
            confidence_score = self.confidence_scores.get(entity1, 0.5)
            item['confidence_score'] = torch.tensor(confidence_score, dtype=torch.float)

        return item

    def get_label(self, entity1, relation, entity2):
        # If rating is greater than 3, label as 1 (positive), else 0 (negative)
        if relation == 'rated_as' and float(entity2) > 3:
            return 1.0
        else:
            return 0.0
        
    

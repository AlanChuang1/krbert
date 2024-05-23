import argparse
import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from data_processing import load_reviews, load_metadata, create_triplets, TripletDataset, calculate_confidence_scores
from model import KRBERTModel
from tqdm import tqdm
import random
from sklearn.metrics import precision_score, recall_score, f1_score


# Define the paths to the dataset files
dataset_folder = os.path.join(os.path.dirname(__file__), 'dataset')
review_file = os.path.join(dataset_folder, 'All_Beauty.jsonl')
meta_file = os.path.join(dataset_folder, 'meta_All_Beauty.jsonl')

def main(args):
    # Load datasets
    reviews_df = load_reviews(review_file)
    metadata_df = load_metadata(meta_file)

    # Create triplets
    triplets = create_triplets(reviews_df, metadata_df)

    print(triplets.size())

    # Calculate confidence scores if required
    confidence_scores = None
    if args.use_confidence:
        confidence_scores = calculate_confidence_scores(triplets)

    sample_size = 12000  
    sampled_triplets = random.sample(triplets, sample_size)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Prepare dataset and dataloader with sampled triplets
    triplet_dataset = TripletDataset(sampled_triplets, tokenizer, confidence_scores, args.use_confidence)
    triplet_dataloader = DataLoader(triplet_dataset, batch_size=32, shuffle=True)

    # Initialize the model
    relation_vocab_size = len(triplet_dataset.relation_vocab) 
    embedding_dim = 64  
    model = KRBERTModel('bert-base-uncased', relation_vocab_size, embedding_dim, args.use_confidence)

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)  # Use PyTorch's AdamW
    epochs = 4
    total_steps = len(triplet_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.BCEWithLogitsLoss().to(device)  

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_labels = []
        all_preds = []
        for step, batch in enumerate(tqdm(triplet_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            relation_ids = batch['relation'].to(device)  # Convert relation to ids
            labels = batch['label'].to(device)  

            if args.use_confidence:
                confidence_scores = batch['confidence_score'].to(device)
                outputs = model(input_ids, attention_mask, relation_ids, confidence_scores)
            else:
                outputs = model(input_ids, attention_mask, relation_ids)

            loss = loss_fn(outputs.squeeze(), labels)
            total_loss += loss.item()

            # Collect true labels and predicted labels
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy() > 0.5)

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(triplet_dataloader)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        print(f"Epoch {epoch+1} completed. Average Loss: {avg_train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KR-BERT model with or without confidence scoring")
    parser.add_argument("--use_confidence", action="store_true", help="Use confidence scoring during training")
    args = parser.parse_args()
    main(args)
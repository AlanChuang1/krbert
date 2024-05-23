import torch
import torch.nn as nn
from transformers import BertModel

class KRBERTModel(nn.Module):
    def __init__(self, bert_model_name, relation_vocab_size, embedding_dim, use_confidence=False):
        super(KRBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.relation_embeddings = nn.Embedding(relation_vocab_size, embedding_dim)
        self.fc = nn.Linear(self.bert.config.hidden_size + embedding_dim, self.bert.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)  
        self.use_confidence = use_confidence

    def forward(self, input_ids, attention_mask, relation_ids, confidence_scores=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls_output = bert_outputs[1]  # [CLS] token output

        relation_embedding = self.relation_embeddings(relation_ids)
        combined = torch.cat((bert_cls_output, relation_embedding), dim=1)

        x = self.fc(combined)
        x = self.dropout(x)
        logits = self.out(x)
        
        if self.use_confidence and confidence_scores is not None:
            logits = logits * confidence_scores.unsqueeze(1)
        
        return logits

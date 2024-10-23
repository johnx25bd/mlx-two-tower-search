import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F

class DocumentDataset(Dataset):
    def __init__(self, df_input):
        self.documents = df_input['document_tokens']
        self.queries = df_input['query_tokens']
        self.labels = df_input['relevance']

    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        return (
            # This outputs tensors of token indices — variable length
                # NOT embeddings
            torch.tensor(self.documents.iloc[idx], dtype=torch.long),
            torch.tensor(self.queries.iloc[idx], dtype=torch.long),
            # This will be a scalar float
            torch.tensor(self.labels.iloc[idx], dtype=torch.float32),
        )

class TwoTowerModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, embedding_layer):
        super(TwoTowerModel, self).__init__()
        
        self.embedding = embedding_layer

        self.doc_rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)

        self.query_rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
    
    def doc_forward(self, doc_ids):
        doc_embed = self.embedding(doc_ids)
        _, doc_hidden = self.doc_rnn(doc_embed)
        doc_encoded = doc_hidden[-1] # average across timesteps
        return doc_encoded

    def query_forward(self, query_ids):
        query_embed = self.embedding(query_ids)
        _, query_hidden = self.query_rnn(query_embed)
        query_encoded = query_hidden[-1]
        return query_encoded
        
    
    def forward(self, doc_ids, query_ids, doc_encoding=None):
        if doc_encoding is None:
            d = self.doc_forward(doc_ids)
        else:
            d = doc_encoding
        q = self.query_forward(query_ids)
        
        similarity = F.cosine_similarity(d, q, dim=1)
        return similarity

    # Loss
        # s = cosine_similarity(hd_n, hq_n)
        # if relevance == 0, s should be low
        # if relevance == 1 or 2, s should be high
        
    # Backprop



__all__ = ['DocumentDataset', 'TwoTowerModel']
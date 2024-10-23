import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F

class DocumentDataset(Dataset):
    def __init__(self, df_input):
        self.docs_rel = df_input['doc_rel_tokens']
        self.docs_irr = df_input['doc_irr_tokens']
        self.queries = df_input['query_tokens']
        # self.labels = df_input['relevance']

    def __len__(self):
        return len(self.docs_rel)
    
    def __getitem__(self, idx):
        return (
            # This outputs tensors of token indices — variable length
                # NOT embeddings
            torch.tensor(self.docs_rel.iloc[idx], dtype=torch.long),
            torch.tensor(self.docs_irr.iloc[idx], dtype=torch.long),
            torch.tensor(self.queries.iloc[idx], dtype=torch.long),
            # This will be a scalar float
            # torch.tensor(self.labels.iloc[idx], dtype=torch.float32),
        )

def loss_fn(rel_similarity, irr_similarity, margin):
    assert rel_similarity.shape == irr_similarity.shape, "Similarity tensors must have the same shape"
    # losses = torch.clamp(margin - rel_similarity + irr_similarity, min=0)
    loss = F.relu(margin - rel_similarity + irr_similarity).mean()
    return loss
    # return torch.max(torch.tensor(0), rel_similarity - irr_similarity + margin)


class TwoTowerModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, embedding_layer, margin):
        super(TwoTowerModel, self).__init__()
        
        self.embedding = embedding_layer
        self.doc_rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.query_rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.margin = margin
    
    def doc_forward(self, doc_ids, doc_mask=None):
        # IS it a lot faster to prepare embeddings prior to training?
        doc_embed = self.embedding(doc_ids)
        doc_lengths = doc_mask.sum(dim=1).cpu() if doc_mask is not None else None
        
        if doc_lengths is not None and torch.any(doc_lengths == 0):
            print("Warning: doc_lengths contains zero values")
            doc_lengths = doc_lengths.clamp(min=1)  # Ensure no zero lengths
        
        doc_packed = nn.utils.rnn.pack_padded_sequence(doc_embed, doc_lengths, batch_first=True, enforce_sorted=False)
        _, doc_hidden = self.doc_rnn(doc_packed)
        doc_encoding = doc_hidden[-1]
        return doc_encoding

    def query_forward(self, query_ids, query_mask=None):
        query_embed = self.embedding(query_ids)
        query_lengths = query_mask.sum(dim=1).cpu() if query_mask is not None else None
        
        if query_lengths is not None and torch.any(query_lengths == 0):
            print("Warning: query_lengths contains zero values")
            query_lengths = query_lengths.clamp(min=1)  # Ensure no zero lengths
        
        query_packed = nn.utils.rnn.pack_padded_sequence(query_embed, query_lengths, batch_first=True, enforce_sorted=False)
        _, query_hidden = self.query_rnn(query_packed)
        query_encoding = query_hidden[-1]
        return query_encoding
        
    
    def forward(self, doc_ids, query_ids, doc_mask=None, query_mask=None):
        
        d = self.doc_forward(doc_ids, doc_mask)
        q = self.query_forward(query_ids, query_mask)
        
        similarity = F.cosine_similarity(d, q, dim=1)
 
        return similarity

    # Loss
        # s = cosine_similarity(hd_n, hq_n)
        # if relevance == 0, s should be low
        # if relevance == 1 or 2, s should be high
        
    # Backprop



__all__ = ['DocumentDataset', 'TwoTowerModel']

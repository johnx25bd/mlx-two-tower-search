import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from utils import str_to_tokens

class DocumentDataset(Dataset): # for training the two-tower model
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
            torch.tensor(self.docs_rel.iloc[idx], dtype=torch.long),
            torch.tensor(self.docs_irr.iloc[idx], dtype=torch.long),
            torch.tensor(self.queries.iloc[idx], dtype=torch.long),
        )

class DocDataset(Dataset): # for creating document projection embeddings
    def __init__(self, df, word_to_idx):
        self.df = df
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        doc = self.df.iloc[idx]['doc_relevant']
        tokens = torch.tensor(str_to_tokens(doc, self.word_to_idx))
        return tokens, idx

def collate_docdataset(batch):
    tokens, indices = zip(*batch)
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    mask = (tokens != 0).float()
    return tokens, mask, torch.tensor(indices)




def loss_fn(rel_similarity, irr_similarity, margin):
    assert rel_similarity.shape == irr_similarity.shape, "Similarity tensors must have the same shape"
    loss = F.relu(margin - rel_similarity + irr_similarity).mean()
    return loss


class DocumentProjection(nn.Module):
    def __init__(self, input_size, embedding_dim):
        super(DocumentProjection, self).__init__()
        self.fc1 = nn.Linear(input_size, embedding_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, doc_encoding):
        x = self.fc1(doc_encoding)
        x = self.relu1(x)
        x = self.fc2(x)

        return x

class QueryProjection(nn.Module):
    def __init__(self, encoding_dim, projection_dim):
        super(QueryProjection, self).__init__()
        self.fc1 = nn.Linear(encoding_dim, encoding_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(encoding_dim, projection_dim)


    def forward(self, doc_encoding):
        x = self.fc1(doc_encoding)
        x = self.relu1(x)
        x = self.fc2(x)

        return x

class TwoTowerModel(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 embedding_layer, # avg pooled embeddings will have dim = embedding_dim
                 projection_dim,
                 margin):
        super().__init__()
        
        self.embedding = embedding_layer
        self.encoding_dim = embedding_dim # for this model, since we're doing avg pooling
        self.doc_projection = DocumentProjection(embedding_dim, projection_dim)
        self.query_projection = QueryProjection(embedding_dim, projection_dim)
        self.margin = margin
    


    def doc_encode(self, doc_ids, doc_mask=None):

        doc_embed = self.embedding(doc_ids)
        doc_embed = doc_embed * doc_mask.unsqueeze(-1) if doc_mask is not None else doc_embed
        doc_encoding = doc_embed.mean(dim=1)
        return doc_encoding

    def query_encode(self, query_ids, query_mask=None):
        query_embed = self.embedding(query_ids)
        query_embed = query_embed * query_mask.unsqueeze(-1) if query_mask is not None else query_embed
        query_encoding = query_embed.mean(dim=1)
        return query_encoding

    def doc_project(self, doc_encoding):
        # Project doc encoding to a lower-dimensional space
        return self.doc_projection(doc_encoding)

    
    def query_project(self, query_encoding):
        return self.query_projection(query_encoding)
    
    def forward(self, doc_ids, query_ids, doc_mask=None, query_mask=None, return_projections=False):
        
        d_encoding = self.doc_encode(doc_ids, doc_mask)
        q_encoding = self.query_encode(query_ids, query_mask)
        
        d_projection = self.doc_project(d_encoding)
        q_projection = self.query_project(q_encoding)
        
        similarity = F.cosine_similarity(d_projection, q_projection, dim=1)
        
        if return_projections:
            return similarity, d_projection, q_projection
        
        return similarity



__all__ = ['DocumentDataset', 'TwoTowerModel']

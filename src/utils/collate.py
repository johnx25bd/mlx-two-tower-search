import torch
from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    docs, queries, labels = zip(*batch)
    docs = pad_sequence(docs, batch_first=True, padding_value=0)
    queries = pad_sequence(queries, batch_first=True, padding_value=0)

    # Create masks
    doc_mask = (docs != 0).float()
    query_mask = (queries != 0).float()
    labels = torch.tensor(labels, dtype=torch.float32)
    return docs, queries, doc_mask, query_mask, labels
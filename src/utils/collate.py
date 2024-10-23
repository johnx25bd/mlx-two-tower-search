import torch
from torch.nn.utils.rnn import pad_sequence

def collate(batch):

    try:
        docs_rel, docs_irr, queries = zip(*batch)
        # print(f"docs_rel shape: {docs_rel[0].shape}")
        # print(f"docs_irr shape: {docs_irr[0].shape}")
        # print(f"queries shape: {queries[0].shape}")

        docs_rel = pad_sequence(docs_rel, batch_first=True, padding_value=0)
        docs_irr = pad_sequence(docs_irr, batch_first=True, padding_value=0)
        queries = pad_sequence(queries, batch_first=True, padding_value=0)

        # print(f"docs_rel shape: {docs_rel.shape}")
        # print(f"docs_irr shape: {docs_irr.shape}")
        # print(f"queries shape: {queries.shape}")

        # Create masks one by one
        # print("Creating docs_rel_mask...")
        docs_rel_mask = (docs_rel != 0).float()
        # print("docs_rel_mask created successfully")

        # print("Creating docs_irr_mask...")
        docs_irr_mask = (docs_irr != 0).float()
        # print("docs_irr_mask created successfully")

        # print("Creating query_mask...")
        query_mask = (queries != 0).float()
        # print("query_mask created successfully")

        return docs_rel, docs_irr, queries, docs_rel_mask, docs_irr_mask, query_mask
    except Exception as e:
        print(f"Error in collate function: {e}")
        print(f"Batch contents: {batch}")
        raise
import torch

def get_docs(embedding: torch.Tensor):
    # load document_embedding_index
    # lookup query embedding in document_embedding_index

    # return top/bottom docs as a list of strings
    rel_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]

    rel_docs_sim = [0.8, 0.7, 0.6, 0.5, 0.4]

    irel_docs = ["doc6", "doc7", "doc8", "doc9", "doc10"]

    irel_docs_sim = [0.3, 0.2, 0.1, 0.09, 0.08]

    return rel_docs, rel_docs_sim, irel_docs, irel_docs_sim

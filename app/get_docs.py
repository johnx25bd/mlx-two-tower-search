import torch
from .preprocess import preprocess


def get_docs(embedding: torch.Tensor):
    # lookup embedding in index
    # comparison with
    # return top docs as a list of strings

    relevant_docs = [
        "doc1",
        "doc2",
        "doc3",
        "doc4",
        "doc5",
        "doc6",
        "doc7",
        "doc8",
        "doc9",
        "doc10",
    ]

    similarity = [0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

    return relevant_docs, similarity

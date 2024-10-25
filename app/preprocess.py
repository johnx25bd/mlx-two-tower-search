import torch
from .utils.preprocess_str import preprocess_query


def preprocess(query: str):
    # preprocess function

    # get_embedding function
    # query_embedding = torch.tensor(query)
    # generate random tensor of dimension 100
    query_embedding = torch.rand(100)

    return query_embedding

import torch
from .utils.load_data import load_word2vec


class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.vocab, cls._instance.embeddings, cls._instance.word_to_idx = (
                load_word2vec()
            )
        return cls._instance

    @classmethod
    def get_model(cls):
        return cls()


def get_string_embedding(input_string, vocab, word_to_idx, embeddings):
    from utils.preprocess_str import preprocess_query

    processed_words = preprocess_query(input_string)
    word_embeddings = []

    for word in processed_words:
        if word in word_to_idx:
            word_idx = word_to_idx[word]
            word_embedding = embeddings[word_idx]
            word_embeddings.append(word_embedding)

    if not word_embeddings:
        return torch.zeros(embeddings.shape[1])

    string_embedding = torch.stack(word_embeddings).mean(dim=0)
    return string_embedding

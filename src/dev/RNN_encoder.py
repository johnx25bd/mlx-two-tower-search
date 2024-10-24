import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_layers=1, bidirectional=False
    ):
        super(RNNEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Define the embedding layer using input_dim for vocab size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # Use input_dim instead of vocab_size
            embedding_dim=embedding_dim,
        )

        # Define the RNN layer
        self.rnn = nn.RNN(
            input_size=embedding_dim,  # The input size is now the embedding dimension
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, token_indices, lengths):
        token_embeddings = self.embedding(token_indices)

        packed_input = pack_padded_sequence(
            token_embeddings, lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, hidden = self.rnn(packed_input)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        if self.bidirectional:
            final_hidden_state = torch.cat(
                (hidden[-2], hidden[-1]), dim=1
            )  # for bidirectional
        else:
            final_hidden_state = hidden[-1]  # for unidirectional

        return final_hidden_state

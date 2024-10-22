import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(RNNEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Define the RNN layer
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.hidden_dim,
        )

        # Forward propagate through RNN
        out, hidden = self.rnn(x, h0)

        # Return the hidden state as the document embedding
        # For simplicity, take the last layer's hidden state as the document embedding
        if self.bidirectional:
            # If bidirectional, concatenate the forward and backward hidden states
            embedding = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            embedding = hidden[-1]

        return embedding


if __name__ == "__main__":
    # Define parameters
    input_dim = 300
    hidden_dim = 128
    num_layers = 1
    bidirectional = True

    # Create an instance of the RNNEncoder
    rnn_encoder = RNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )

    # Create a batch of documents (batch_size, sequence_length, input_dim)
    batch_size = 4
    sequence_length = 10
    x = torch.randn(batch_size, sequence_length, input_dim)

    # Get document embeddings
    embeddings = rnn_encoder(x)
    print("Document Embeddings Shape:", embeddings.shape)

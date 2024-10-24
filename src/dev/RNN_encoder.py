import torch
import torch.nn as nn

class RNNEncoder(nn.Module):

    def __init__(self, embedding_lookup, hidden_dim, num_layers=1):
        super(RNNEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        vocab_size, embedding_dim = embedding_lookup.shape

        # Shared embedding layer
        self.embedding = nn.Embedding.from_pretrained(embedding_lookup, freeze=False)

        # Separate vanilla RNNs for query and documents
        self.queryRNN = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.docRNN = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query_indices,
        query_lengths,
        pos_doc_indices,
        pos_doc_lengths,
        neg_doc_indices,
        neg_doc_lengths,
    ):
        # Get embeddings for query and documents
        query_embeddings = self.embedding(
            query_indices
        )  # shape: (batch_size, seq_len, embedding_dim)
        pos_doc_embeddings = self.embedding(pos_doc_indices)
        neg_doc_embeddings = self.embedding(neg_doc_indices)

        # Pack sequences to handle variable lengths
        query_packed = nn.utils.rnn.pack_padded_sequence(
            query_embeddings, query_lengths, batch_first=True, enforce_sorted=False
        )
        pos_doc_packed = nn.utils.rnn.pack_padded_sequence(
            pos_doc_embeddings, pos_doc_lengths, batch_first=True, enforce_sorted=False
        )
        neg_doc_packed = nn.utils.rnn.pack_padded_sequence(
            neg_doc_embeddings, neg_doc_lengths, batch_first=True, enforce_sorted=False
        )

        # Pass embeddings through the respective vanilla RNNs
        query_packed_output, query_hidden = self.queryRNN(query_packed)
        pos_doc_packed_output, pos_doc_hidden = self.docRNN(pos_doc_packed)
        neg_doc_packed_output, neg_doc_hidden = self.docRNN(neg_doc_packed)

        # Unpack the sequences to get back to the padded sequence format (if needed for further processing)
        query_output, _ = nn.utils.rnn.pad_packed_sequence(
            query_packed_output, batch_first=True
        )
        pos_doc_output, _ = nn.utils.rnn.pad_packed_sequence(
            pos_doc_packed_output, batch_first=True
        )
        neg_doc_output, _ = nn.utils.rnn.pad_packed_sequence(
            neg_doc_packed_output, batch_first=True
        )

        # Layer normalization (optional)
        query_hidden_normalized = self.layer_norm(
            query_hidden[-1]
        )  # Take the last layer's hidden state
        pos_doc_hidden_normalized = self.layer_norm(pos_doc_hidden[-1])
        neg_doc_hidden_normalized = self.layer_norm(neg_doc_hidden[-1])

        # Return the final hidden states
        return (
            query_hidden_normalized,
            pos_doc_hidden_normalized,
            neg_doc_hidden_normalized,
        )

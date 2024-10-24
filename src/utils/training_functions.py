import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


def prepare_dataloader(indexes, batch_size, padding_value=0.0):

    # Extract and filter out sequences with length 0
    filtered_data = [
        (q, r, i)
        for q, r, i in zip(
            indexes["query_indexes"],
            indexes["relevant_doc_indexes"],
            indexes["irrelevant_doc_indexes"],
        )
        if len(q) > 0 and len(r) > 0 and len(i) > 0
    ]

    # Unzip the filtered data back into individual lists
    query_idx_filtered, rel_doc_idx_filtered, irrel_doc_idx_filtered = zip(
        *filtered_data
    )

    def process_and_pad(sequences):
        lengths = [len(seq) for seq in sequences]
        sequences_tensors = [torch.tensor(seq) for seq in sequences]
        padded_sequences = pad_sequence(
            sequences_tensors, batch_first=True, padding_value=padding_value
        )
        return padded_sequences, torch.tensor(lengths)

    query_idx_padded, query_lengths = process_and_pad(query_idx_filtered)
    rel_doc_idx_padded, rel_lengths = process_and_pad(rel_doc_idx_filtered)
    irrel_doc_idx_padded, irrel_lengths = process_and_pad(irrel_doc_idx_filtered)

    dataset = TensorDataset(
        query_idx_padded,
        query_lengths,
        rel_doc_idx_padded,
        rel_lengths,
        irrel_doc_idx_padded,
        irrel_lengths,
    )

    training_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return training_data


def compute_triplet_loss(query_encoded, relevant_encoded, irrelevant_encoded, margin):
    relevant_distances = 1 - F.cosine_similarity(query_encoded, relevant_encoded, dim=1)
    irrelevant_distances = 1 - F.cosine_similarity(
        query_encoded, irrelevant_encoded, dim=1
    )
    triplet_loss = F.relu(relevant_distances - irrelevant_distances + margin).mean()
    return triplet_loss


def save_model(epoch, save_path):
    torch.save(
        {
            "query_encoder_state_dict": query_encoder.state_dict(),
            "doc_encoder_state_dict": doc_encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )

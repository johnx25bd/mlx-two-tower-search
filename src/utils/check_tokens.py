import gensim

# Path to the pre-trained model (Google News word2vec binary file)
model_path = "some path"

# Load the model - not sure if this is correct
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

# import tokens from training data 


import gensim

# Load the Google News Word2Vec model
model_path = "GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)


def check_token_in_vocab(tokens, model):
    not_in_vocab = []
    count_not_in_vocab = 0

    for token in tokens:
        if token not in model.key_to_index:
            not_in_vocab.append(token)
            count_not_in_vocab += 1

    # Print the count and list of tokens not in the model vocabulary
    print(f"Number of tokens not in the model vocabulary: {count_not_in_vocab}")
    print("List of tokens not in the vocabulary:")
    # Save the list of tokens not found in vocabulary to a file
    with open("tokens_not_in_vocab.txt", "w") as f:
        for token in not_in_vocab:
            f.write(token + "\n")


# (replace with your actual tokens)
tokens = ["Apple", "apple", "U.S.A.", "co-founder", "running", "ten", "10", "thisisnotaword"]

check_token_in_vocab(tokens, model)




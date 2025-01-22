import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier


def find_optimal_clusters(data, range_n_clusters):
    """Find the optimal number of clusters using silhouette score."""
    best_n_clusters = None
    best_score = -1
    best_model = None

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_model = kmeans

    return best_n_clusters, best_model


def train_tokenizer(data, range_n_clusters, knn_neighbors=5):
    """Train a tokenizer using K-means and k-NN."""
    # Step 1: Find optimal number of clusters
    n_clusters, kmeans = find_optimal_clusters(data, range_n_clusters)
    print(f"Optimal number of clusters: {n_clusters}")

    # Step 2: Assign tokens to data points
    tokens = kmeans.predict(data)
    print(f"Tokens assigned to data points: {tokens[:5]}")  # Print first 5 token assignments

    # Step 3: Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
    knn.fit(data, tokens)

    return kmeans, knn, tokens


def tokenize_new_data(knn, new_data):
    """Tokenize new time series data using the trained k-NN classifier."""
    return knn.predict(new_data)


def returnData(abspath, filepath_data):
    totalpath_data = abspath + filepath_data
    data = pd.read_csv(totalpath_data, header=None)
    return data

def returnLabels(abspath, filepath_labels):
    totalpath_labels = abspath + filepath_labels

    # This was needed when labels were the first column of my csv
  #  labels = pd.read_csv(totalpath_labels, header=None)[:][0]

    labels = pd.read_csv(totalpath_labels, header=None)
    if type(labels) != type(pd.Series):
        labels = labels.iloc[:, 0]  # convert to series
    print(type(labels))

    return labels

def makeLabelsInt(labels):
    # print(labels)
    return labels.map({'C': 0, 'D': 1}).to_numpy()



# Example Usage
if __name__ == "__main__":
    abspath = "/home/vang/Downloads/"
    filepath_data = "Lu_sR50_2025-01-06_01-40-21_output (Copy).csv"
    filepath_data = "Pitt_sR11025.0_2025-01-20_23-11-13_output.csv"
    data = returnData(abspath, filepath_data)

    filepath_labels = "Lu_sR50_2025-01-06_01-40-21_output.csv"
    filepath_labels = "Pitt_sR11025.0_2025-01-20_23-12-07_labels.csv"
    initial_labels = returnLabels(abspath, filepath_labels)

    # Drop NaN rows from data
    data = data.dropna()
    # Reset indices after dropping rows
    data = data.reset_index(drop=True)
    # Ensure labels align with the updated data
    labels = initial_labels[data.index]
    labels = labels.reset_index(drop=True)

    labels = makeLabelsInt(labels)

    n_clusters_min = 40 # Was initially 2
    n_clusters_max = 50 # Was initially 10
    # Define the range for the number of clusters
    range_n_clusters = range(n_clusters_min, n_clusters_max)  # Desirable range

    # Train the tokenizer
    knn_neighbors = 50
    kmeans_model, knn_model, tokens = train_tokenizer(data, range_n_clusters, knn_neighbors = knn_neighbors)

 #   corpus = [knn_model.predict(ts.reshape(1, -1))[0] for ts in data]
 #   print(f"Corpus: {corpus[:5]}")  # Display first 5 sequences

    # Create the corpus as a sequence of tokens
    corpus = tokens.tolist()  # List of tokens representing the time series sequence
    print(f"Corpus (first 5 tokens): {corpus[:5]}")

    # Example of a token sequence
    token_sequence = [token for token in corpus]
    print(f"Token Sequence: {token_sequence[:50]}")


    # Reshaping the array into 10 sequences of 5 elements each
    sequences = [token_sequence[i:i + 5] for i in range(0, len(token_sequence), 5)]

    # Print the sequences
    print(sequences)




    # Generate the corpus
  #  corpus = [knn_model.predict(ts.reshape(1, -1)).tolist() for ts in data]
 #   print(f"Corpus: {corpus[:5]}")  # Display first 5 sequences
 #   print(corpus)

 #   token_sequence = [token[0] for token in corpus]
#    print(f"Token Sequence: {token_sequence}")



    window_size = 20  # Define the window size for context
    target_context_pairs = []

    for i in range(window_size, len(corpus) - window_size):
        target = corpus[i]
        context = corpus[i - window_size:i] + corpus[i + 1:i + window_size + 1]
        for ctx in context:
            target_context_pairs.append((target, ctx))

    print(f"Generated target-context pairs: {target_context_pairs[:5]}")








    # ----- -----     SKIP GRAM     ----- -----
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Dense, Flatten
    from tensorflow.keras.preprocessing.sequence import skipgrams
    from tensorflow.keras.preprocessing.text import Tokenizer

    def create_skipgram_pairs(sequence, vocab_size, window_size=2):
        """
        Generate skip-gram pairs using TensorFlow's skipgrams utility.
        """
        pairs, labels = skipgrams(sequence, vocabulary_size=vocab_size, window_size=window_size)
        return np.array(pairs), np.array(labels)

    def build_skipgram_model(vocab_size, embedding_dim):
        """
        Build a skip-gram model with a single hidden layer.
        """
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1),
            Flatten(),
            Dense(vocab_size, activation='softmax')  # Output layer for token prediction
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model

    def train_skipgram(corpus, vocab_size, embedding_dim=50, window_size=2, epochs=10):
        """
        Train a skip-gram model on the tokenized corpus.
        """
        # Flatten the corpus for skip-gram generation
        flat_corpus = [token for sequence in corpus for token in sequence]

        # Generate skip-gram pairs
        pairs, labels = create_skipgram_pairs(flat_corpus, vocab_size, window_size)

        # Build the skip-gram model
        model = build_skipgram_model(vocab_size, embedding_dim)

        # Train the model
        pairs_context, pairs_target = pairs[:, 0], pairs[:, 1]
        model.fit(pairs_context, pairs_target, epochs=epochs, batch_size=256, verbose=1)

        return model

    # Flatten the corpus
  #  tokenized_data = [item[0] for item in corpus]  # Extract the token from each sublist
 #   print(f"Tokenized Data: {tokenized_data[:5]}")  # Show first 5 tokens

    # Create sequences from the tokenized data (e.g., sequences of length 5)
 #   sequences = [tokenized_data[i:i + 5] for i in range(0, len(tokenized_data), 5)]
 #   print(f"Tokenized Sequences: {sequences[:3]}")  # Display first 3 sequences

    tokenized_data = [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [1, 3, 4, 6, 7]
    ]

    tokenized_data = sequences


    # Parameters
    vocab_size = n_clusters_max # Set vocabulary size based on your tokens
    embedding_dim = 50
    window_size = 20
    epochs = 10

    # Train skip-gram model
    model = train_skipgram(tokenized_data, vocab_size, embedding_dim, window_size, epochs)
    print("Skip-gram model trained!")

    import numpy as np


    def get_all_embeddings(model, sequences):
        """
        Get embeddings for all tokens in all sequences using the trained skip-gram model.

        Parameters:
        - model: Trained skip-gram model.
        - sequences: List of sequences, where each sequence is a list of tokens (integers).

        Returns:
        - all_embeddings: A dictionary where keys are tokens and values are their embeddings.
        """
        # Extract the weights of the embedding layer
        embedding_layer = model.layers[0]  # The first layer is the embedding layer
        embeddings = embedding_layer.get_weights()[0]  # Get the embedding matrix

        # Map each token to its embedding
        all_embeddings = {token: embeddings[token] for sequence in sequences for token in sequence}

        return all_embeddings


    # Assume `sequences` is your input data and `model` is the trained skip-gram model
    time_series_embeddings = get_all_embeddings(model, sequences)

    # Print embeddings for all tokens
    for token, embedding in time_series_embeddings.items():
        print(f"Token {token}: Embedding {embedding}")


    def get_sequence_embeddings(sequences, model):
        """
        Given a list of sequences, get the average embedding for each sequence
        by averaging the embeddings of its tokens.
        """
        # Initialize an empty list to store the sequence embeddings
        sequence_embeddings = []

        for sequence in sequences:
            # Get embeddings for each token in the sequence
            embeddings = np.array([model.layers[0].get_weights()[0][token] for token in sequence])

            # Compute the average embedding for the sequence
            sequence_embedding = np.mean(embeddings, axis=0)

            # Append the average embedding to the list
            sequence_embeddings.append(sequence_embedding)

        # Convert the list of sequence embeddings into a NumPy array
        return np.array(sequence_embeddings)


    # Function to get embeddings for each sequence directly
    def get_sequence_embedding(token_sequence, model):
        """
        Given a sequence of tokens, get the embedding for the entire sequence by averaging the embeddings
        of all tokens in the sequence.
        """
        embeddings = []

        # Get embedding for each token in the sequence
        for token in token_sequence:
            # Get the embedding for the token from the embedding layer
            token_embedding = model.layers[0].get_weights()[0][token]  # Embedding layer's weights
            embeddings.append(token_embedding)

        # Convert to a numpy array and calculate the mean across all tokens in the sequence
        embeddings = np.array(embeddings)
        sequence_embedding = np.mean(embeddings, axis=0)

        return sequence_embedding

    # Now get the embeddings for each sequence
 #   time_series_embeddings = get_sequence_embedding(token_sequence, model)
    # Assuming 'model' is the trained skipgram model
    # Get embeddings for all sequences
    time_series_embeddings = np.array([get_sequence_embedding(seq, model) for seq in sequences])

    print(f"Shape of sequences: {len(sequences)}")
    # Show the result
    print(f"Shape of sequence embedding: {time_series_embeddings.shape}")
    print(f"Sequence embedding:\n{time_series_embeddings}")


    import time
    from datetime import datetime
    def SaveEmbeddingsToOutput(embeddings):
        formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        df = pd.DataFrame(embeddings)
        filename = abspath + "Embeddings" + "_" + formatted_datetime + ".csv"

        # Writing to CSV with pandas (which is generally faster)
        df.to_csv(filename, index=False, header=False)

        df_labels = pd.DataFrame(labels)
        filename = abspath + "Labels" + "_" + formatted_datetime + ".csv"
        df_labels.to_csv(filename, index=False, header=False)
        pass;


    SaveEmbeddingsToOutput(time_series_embeddings)

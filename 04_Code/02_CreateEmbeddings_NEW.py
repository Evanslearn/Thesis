import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing.sequence import skipgrams

from utils00 import (
    makeLabelsInt,
    doTrainValTestSplit,
    doTrainValTestSplit2,
    readCsvAsDataframe
)

def find_optimal_clusters(data, range_n_clusters):
    """Find the optimal number of clusters using silhouette score."""
    best_n_clusters, best_score, best_model  = None, -1, None

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=config["random_state"])
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        print(f"Clusters = {n_clusters} -> Silhouette score = {score}")
        if score > best_score:
            best_n_clusters, best_score, best_model = n_clusters, score, kmeans

    return best_n_clusters, best_model

def train_tokenizer(data, range_n_clusters, knn_neighbors=5):
    """Train a tokenizer using K-means and k-NN."""
    n_clusters, kmeans = find_optimal_clusters(data, range_n_clusters)
    print(f"\nOptimal number of clusters: {n_clusters}")

    tokens = kmeans.predict(data)
    print(f"Tokens assigned to first 5 data points: {tokens[:5]}")  # Print first 5 token assignments

    # Step 3: Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors).fit(data, tokens)
    return kmeans, knn, tokens, n_clusters

def create_sequences(token_sequence, window_size, stride):
    return [token_sequence[i:i + window_size] for i in range(0, len(token_sequence) - window_size + 1, stride)]

# ----- -----     SKIP GRAM     ----- -----
def generate_skipgram_pairs(sequence, vocab_size, window_size=2, negative_samples=5):
    """
    Generate skip-gram pairs using TensorFlow's skipgrams utility.
    """
    pairs, labels = skipgrams(
        sequence,
        vocabulary_size=vocab_size,
        window_size=window_size,
        negative_samples= negative_samples # Needed by NCE
    )
    return np.array(pairs), np.array(labels)

def build_skipgram_model(vocab_size, embedding_dim, loss = "sparse_categorical_crossentropy"):
    """    Build a skip-gram model with a single hidden layer.    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1),
        Flatten(),
        Dense(vocab_size, activation='softmax')  # Output layer for token prediction
    ])
    model.compile(optimizer='adam', loss=loss)
    return model

def train_skipgram(corpus, vocab_size, embedding_dim=50, window_size=2, epochs=10, loss= "sparse_categorical_crossentropy"):
    """    Train a skip-gram model on the tokenized corpus.    """
    # Flatten the corpus for skip-gram generation
    flat_corpus = [token for sequence in corpus for token in sequence]

    # Generate skip-gram pairs
    pairs, labels = generate_skipgram_pairs(flat_corpus, vocab_size, window_size)

    # Build the skip-gram model
    model = build_skipgram_model(vocab_size, embedding_dim, loss)

    # Train the model
    pairs_context, pairs_target = pairs[:, 0], pairs[:, 1]
    model.fit(pairs_context, pairs_target, epochs=epochs, batch_size=config["batch_size"], verbose=1)

    return model

# Function to get embeddings for each sequence directly
def get_sequence_embedding(token_sequence, model):
    """Compute sequence embedding by averaging token embeddings."""
    token_embedding = np.array([model.layers[0].get_weights()[0][token] for token in token_sequence])

    # Convert to a numpy array and calculate the mean across all tokens in the sequence
    sequence_embedding = np.mean(np.array(token_embedding), axis=0)

    return sequence_embedding

def convertBackIntoTokenEmbeddings(token_sequence, sequences, sequence_embeddings):
    # Step 3: Map back to individual tokens
    token_embeddings = np.zeros((len(token_sequence), sequence_embeddings.shape[1]))
    token_counts = np.zeros(len(token_sequence))

    for i, seq in enumerate(sequences):
        for j, token in enumerate(seq):
            token_embeddings[i + j] += sequence_embeddings[i]
            token_counts[i + j] += 1
    token_embeddings /= token_counts[:, None]

    print(f"Shape of sequences: {len(sequences)}")
    print(f"Shape of sequence embedding: {token_embeddings.shape}")
  #  print(f"Sequence embedding:\n{token_embeddings}")
    return token_embeddings

def SaveEmbeddingsToOutput(embeddings, labels, subfolderName, indices=None, setType="NO", **kwargs):
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    case_type = "Pitt" if "Pitt" in filepath_data else "Lu"
    case_type_str = f"{case_type}_{setType}" if setType != "NO" else f"_{case_type}_"

    filename_variables = "".join(f"{key}{value}".replace("{", "").replace("}", "") + "_" for key, value in kwargs.items()).rstrip("_")

    # Helper function to generate paths dynamically
    def generate_path(prefix):
        return f"{subfolderName}/{prefix}_{case_type_str}_{filename_variables}_{formatted_datetime}.csv"

    # Writing to CSV with pandas (which is generally faster)
    pd.DataFrame(embeddings).to_csv(generate_path("Embeddings"), index=False, header=False)
    pd.DataFrame(labels).to_csv(generate_path("Labels"), index=False, header=False)
    # Save indices only if provided
    if indices is not None:
        pd.DataFrame(indices).to_csv(generate_path("Indices"), index=False, header=False)

    return

def plot_tsne(data, labels, title):
    """Applies t-SNE and plots results."""
    tsne = TSNE(n_components=2, perplexity=config["perplexity"], random_state=config["random_state"])
    transformed = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=transformed[:, 0], y=transformed[:, 1], hue=labels, palette="viridis", alpha=0.6)
    plt.title("t-SNE Visualization " + title)
    plt.xlabel("t-SNE Component 1"); plt.ylabel("t-SNE Component 2")
    plt.show()

def print_data_info(data, labels, stage=""):
    print(f"----- {stage} -----")
    print(f"Labels shape = {labels.shape}, Data shape = {data.shape}")

# Configuration dictionary to store hyperparameters and settings
config = {
    "n_clusters_min": 5,        # Min number of clusters for KMeans - 2
    "n_clusters_max": 30,       # Max number of clusters for KMeans - 10
    "knn_neighbors": 50,        # Number of neighbors for k-NN - 50
    "window_size": 10,          # Window size for sequence generation - 10
    "stride": 1,                # Stride for sequence generation - 1
    "embedding_dim": 300,       # Dimension of word embeddings - 300
    "window_size_skipgram": 20, # - 20
    "epochs": 1,                # Number of training epochs
    "batch_size": 256,          # Batch size for training
    "perplexity": 30,           # t-SNE perplexity
    "random_state": 42,         # Random state for reproducibility
    "output_folder": "02_Embeddings"  # Folder for saving embeddings

}

# Example Usage
if __name__ == "__main__":
    timeSeriesDataPath = "/01_TimeSeriesData/"
    folderPath = os.getcwd() + timeSeriesDataPath
    filepath_data = "Lu_sR50_2025-01-06_01-40-21_output (Copy).csv"
    filepath_data = "Pitt_sR11025.0_2025-01-20_23-11-13_output.csv"
    filepath_data = "Pitt_output_sR11025_frameL2048_hopL512_thresh0.02_2025-02-22_14-49-06.csv"

    filepath_labels = "Lu_sR50_2025-01-06_01-40-21_output.csv"
    filepath_labels = "Pitt_sR11025.0_2025-01-20_23-12-07_labels.csv"
    filepath_labels = "Pitt_labels_sR11025_frameL2048_hopL512_thresh0.02_2025-02-22_14-49-06.csv"

    data = readCsvAsDataframe(folderPath, filepath_data)
    initial_labels = readCsvAsDataframe(folderPath, filepath_labels, dataFilename = "labels", as_series=True)

    print_data_info(initial_labels, data, "BEFORE DROPPING NA")
    # Drop NaN rows from data, # Reset indices after dropping rows
    data = data.dropna().reset_index(drop=True)
    # Ensure labels align with the updated data
    labels = initial_labels[data.index]
    labels = labels.reset_index(drop=True)

    labels = makeLabelsInt(labels)
    print_data_info(labels, data, "AFTER DROPPING NA")

    data_train, data_val, data_test, labels_train, labels_val, labels_test, val_ratio, indices_train, indices_val, indices_test = doTrainValTestSplit(data, labels)
    print(f'\nLength of X is = {len(data)}. Length of Y is = {len(labels)}')

    print("data shape = {0}\ndata_train shape = {1}\ndata_val shape = {2}\ndata_test shape = {3}".format(
        data.shape, data_train.shape, data_val.shape, data_test.shape))
    range_n_clusters = range(config["n_clusters_min"], config["n_clusters_max"])  # Desirable range

    # Train the tokenizer
    knn_neighbors = config["knn_neighbors"]
    kmeans_model, knn_model, tokens_train, n_clusters = train_tokenizer(data_train, range_n_clusters, knn_neighbors = knn_neighbors)

    plot_tsne(np.array(data_train), labels_train, "of data_train")
    plot_tsne(data_train, kmeans_model.fit_predict(data_train), f"with {n_clusters} Clusters")

    # Tokenize Train, Val, Test
    train_token_sequence = tokens_train.tolist()
    val_token_sequence = kmeans_model.predict(data_val).tolist()
    test_token_sequence = kmeans_model.predict(data_test).tolist()

    window_size = config["window_size"]  # Length of each sequence
    stride = config["stride"]  # Step size to slide the window (1 ensures maximum overlap)

    train_sequences = create_sequences(train_token_sequence, window_size, stride)
    val_sequences = create_sequences(val_token_sequence, window_size, stride)
    test_sequences = create_sequences(test_token_sequence, window_size, stride)

    # Parameters
    vocab_size = config["n_clusters_max"]  # Set vocabulary size based on your tokens
    embedding_dim = config["embedding_dim"]
    window_size_skipgram = config["window_size_skipgram"]
    epochs = config["epochs"]
    # loss = "nce" # NEEDs to be IMPLEMENTED FROM SCRATCH

    # Train skip-gram model
    skipgram_model = train_skipgram(train_sequences, vocab_size, embedding_dim, window_size_skipgram, epochs)  # , loss)
    print("Skip-gram model trained!")

    # Step 2: Compute sequence embeddings
    train_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in train_sequences])
    val_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in val_sequences])
    test_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in test_sequences])
    print(train_embeddings.shape); print(val_embeddings.shape); print(test_embeddings.shape)
    print(type(train_embeddings))

    train_timeseries_embeddings = convertBackIntoTokenEmbeddings(train_token_sequence, train_sequences, train_embeddings)
    val_timeseries_embeddings = convertBackIntoTokenEmbeddings(val_token_sequence, val_sequences, val_embeddings)
    test_timeseries_embeddings = convertBackIntoTokenEmbeddings(test_token_sequence, test_sequences, test_embeddings)
    trainValTest_embeddings = np.vstack([train_timeseries_embeddings, val_timeseries_embeddings, test_timeseries_embeddings])

    indices_all = np.vstack([indices_train.reshape(-1, 1), indices_val.reshape(-1, 1), indices_test.reshape(-1, 1)])

    name_kwargs = {
        "nCl": n_clusters,
        "nN": knn_neighbors,
        "winSize": window_size,
        "stride": stride,
        "winSizeSkip": window_size_skipgram,
        "nEmbeddings": embedding_dim
    }
    subfoldername = config["output_folder"]

    # ----- MAYBE FIX LIKE THIS -----
    labels_all = np.concatenate([labels_train, labels_val, labels_test])
    SaveEmbeddingsToOutput(trainValTest_embeddings, labels_all, subfoldername, indices_all, **name_kwargs)
  #  SaveEmbeddingsToOutput(trainValTest_embeddings, labels, subfoldername, indices_all, **name_kwargs)

    name_kwargs_train = {
        "train": "Set",
        **name_kwargs
    }
    name_kwargs_val = {
        "val": "Set",
        **name_kwargs
    }
    name_kwargs_test = {
        "test": "Set",
        **name_kwargs
    }
    SaveEmbeddingsToOutput(train_timeseries_embeddings, labels_train, subfoldername, indices_train, **name_kwargs_train)
    SaveEmbeddingsToOutput(val_timeseries_embeddings, labels_val, subfoldername, indices_val, **name_kwargs_val)
    SaveEmbeddingsToOutput(test_timeseries_embeddings, labels_test, subfoldername, indices_test, **name_kwargs_test)
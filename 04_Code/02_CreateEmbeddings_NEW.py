import os
from collections import defaultdict

import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing.sequence import skipgrams

from utils00 import (
    makeLabelsInt,
    doTrainValTestSplit,
    readCsvAsDataframe, plot_tsnePCAUMAP
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
    pairs = np.array(pairs, dtype=np.int32).reshape(-1, 2)
    labels = np.array(labels, dtype=np.int32)
    return pairs, labels

def build_skipgram_model(vocab_size, embedding_dim, loss = "sparse_categorical_crossentropy"):
    """    Build a skip-gram model with a single hidden layer.    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1),
        Flatten(),
        Dense(vocab_size, activation='softmax')  # Output layer for token prediction
    ])
    model.compile(optimizer=config['optimizer_skipgram'], loss=loss)
    return model

def train_skipgram(corpus, vocab_size, embedding_dim=50, window_size=2, epochs=10, loss= "sparse_categorical_crossentropy"):
    """    Train a skip-gram model on the tokenized corpus.    """
    # Flatten the corpus for skip-gram generation
    flat_corpus = [token for sequence in corpus for token in sequence]
    print(f"📏 flat_corpus length: {len(flat_corpus)}")
    print(f"🧠 Unique tokens: {set(flat_corpus)}")
    print("Sample sequences:")
    for seq in corpus[:5]:
        print(seq)

    assert isinstance(flat_corpus[0], (int, np.integer)), "Expected integer token IDs"
    # Generate skip-gram pairs
    pairs, labels = generate_skipgram_pairs(flat_corpus, vocab_size, window_size)

    # Build the skip-gram model
    model = build_skipgram_model(vocab_size, embedding_dim, loss)

    # Train the model
    pairs_context, pairs_target = pairs[:, 0], pairs[:, 1]

    print(f"Context shape: {pairs_context.shape}")
    print(f"Target shape: {pairs_target.shape}")
    print(f"Sample context: {pairs_context[:5]}")
    print(f"Sample target: {pairs_target[:5]}")



    model.fit(pairs_context, pairs_target, epochs=epochs, batch_size=config["batch_size"], verbose=1)

    return model

# Function to get embeddings for each sequence directly
def get_sequence_embedding(token_sequence, model):
    """Compute sequence embedding by averaging token embeddings."""
    token_embedding = np.array([model.layers[0].get_weights()[0][token] for token in token_sequence])

    # ----- COULD TRY THIS INSTEAD OF THE REGULAR MEAN
 #   weights = np.arange(1, len(token_sequence) + 1)  # Linear weight increase
#    sequence_embedding = np.average(token_embedding, axis=0, weights=weights)
    # Convert to a numpy array and calculate the mean across all tokens in the sequence
    sequence_embedding = np.mean(np.array(token_embedding), axis=0)

    return sequence_embedding

def convertBackIntoTokenEmbeddings(token_sequence, sequences, sequence_embeddings):
    token_embeddings = np.zeros((len(token_sequence), sequence_embeddings.shape[1]))
    token_counts = np.zeros(len(token_sequence))

    for i, seq in enumerate(sequences):
        for j, token in enumerate(seq):
            token_embeddings[i + j] += sequence_embeddings[i]
            token_counts[i + j] += 1
    token_embeddings /= token_counts[:, None]

    print(f"Shape of sequences: {len(sequences)}\nShape of sequence embedding: {token_embeddings.shape}")
    return token_embeddings

def returnFormattedDateTimeNow():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def SaveEmbeddingsToOutput(embeddings, labels, subfolderName, formatted_datetime, indices=None, setType="NO", **kwargs):

    case_type = "Pitt" if "Pitt" in filepath_data else "Lu"
    case_type_str = f"{case_type}_{setType}" if setType != "NO" else f"_{case_type}_"

    filename_variables = "".join(f"{key}{value}".replace("{", "").replace("}", "") + "_" for key, value in kwargs.items()).rstrip("_")

    # Helper function to generate paths dynamically
    def generate_path(prefix):
        return f"{subfolderName}/{prefix}_{case_type_str}{filename_variables}_{formatted_datetime}.csv"

    # Writing to CSV with pandas (which is generally faster)
    pd.DataFrame(embeddings).to_csv(generate_path("Embeddings"), index=False, header=False)
    pd.DataFrame(labels).to_csv(generate_path("Labels"), index=False, header=False)
    # Save indices only if provided
    if indices is not None:
        pd.DataFrame(indices).to_csv(generate_path("Indices"), index=False, header=False)

    return

def print_data_info(data, labels, stage=""):
    print(f"----- {stage} -----")
    print(f"Labels shape = {labels.shape}, Data shape = {data.shape}")

# Configuration dictionary to store hyperparameters and settings
config = {
    "n_clusters_min": 5,        # Min number of clusters for KMeans - 2
    "n_clusters_max": 6,       # Max number of clusters for KMeans - 10
    "knn_neighbors": 20,        # Number of neighbors for k-NN - 50
    "window_size": 20,          # Window size for sequence generation - 10
    "stride": 1,                # Stride for sequence generation - 1
    "embedding_dim": 50,       # Dimension of word embeddings - 300
    "window_size_skipgram": 20, # - 20
    "epochs": 1,                # Number of training epochs
    "optimizer_skipgram": 'adam',
    "skipgram_loss": "sparse_categorical_crossentropy", # could also try nce
    "batch_size": 128,          # Batch size for training
    "perplexity": 30,           # t-SNE perplexity
    "random_state": 42,         # Random state for reproducibility
    "output_folder": "02_Embeddings"  # Folder for saving embeddings
}


def slice_timeseries_rowwise(data, window_length, stride):
    segments = []
    origins = []

    for idx, row in data.iterrows():
        row_values = row.drop("index").values  # drop before training
        for i in range(0, len(row_values) - window_length + 1, stride):
            segment = row_values[i:i+window_length]
            segments.append(segment)
            origins.append(idx)

    return np.array(segments), np.array(origins)


# Example Usage
if __name__ == "__main__":
    timeSeriesDataPath = "/01_TimeSeriesData/"
    folderPath = os.getcwd() + timeSeriesDataPath
    filepath_data = "Lu_sR50_2025-01-06_01-40-21_output (Copy).csv"
    filepath_data = "Pitt_sR11025.0_2025-01-20_23-11-13_output.csv"
    filepath_data = "Pitt_output_sR11025_frameL2048_hopL512_thresh0.02_2025-02-22_14-49-06.csv"
    filepath_data = "Pitt_output_sR11025_frameL2048_hopL512_thresh0.0_2025-03-25_17-18-31.csv"

    filepath_labels = "Lu_sR50_2025-01-06_01-40-21_output.csv"
    filepath_labels = "Pitt_sR11025.0_2025-01-20_23-12-07_labels.csv"
    filepath_labels = "Pitt_labels_sR11025_frameL2048_hopL512_thresh0.02_2025-02-22_14-49-06.csv"
    filepath_labels = "Pitt_labels_sR11025_frameL2048_hopL512_thresh0.0_2025-03-25_17-18-31.csv"

    data = readCsvAsDataframe(folderPath, filepath_data)
    initial_labels = readCsvAsDataframe(folderPath, filepath_labels, dataFilename = "labels", as_series=True)

    data["index"] = data.index  # Preserve original row index in a column
    print_data_info(initial_labels, data, "BEFORE DROPPING NA")

    # Drop NaN rows from data, # Reset indices after dropping rows
    data = data.dropna().reset_index(drop=True)
    # Ensure labels align with the updated data
    labels = initial_labels[data.index]
    labels = labels.reset_index(drop=True)

    labels = makeLabelsInt(labels)
    print_data_info(labels, data, "AFTER DROPPING NA")

    _, _, _, _, _, _, val_ratio, indices_train, indices_val, indices_test = doTrainValTestSplit(data, labels)

    # Now use the indices to get real splits — preserving original indices!
    data_train = data.iloc[indices_train].reset_index(drop=False)
    data_val = data.iloc[indices_val].reset_index(drop=False)
    data_test = data.iloc[indices_test].reset_index(drop=False)

    labels_train = pd.Series(labels[indices_train]).reset_index(drop=True)
    labels_val = pd.Series(labels[indices_val]).reset_index(drop=True)
    labels_test = pd.Series(labels[indices_test]).reset_index(drop=True)

    print(data_train.columns)
    print(data_train[0])

    # Drop "index" column before scaling
    X_train_numerical = data_train.drop(columns=["index"])
    X_val_numerical = data_val.drop(columns=["index"])
    X_test_numerical = data_test.drop(columns=["index"])

    # Convert column names to strings (optional, to avoid sklearn type-checking issues)
    X_train_numerical.columns = X_train_numerical.columns.astype(str)
    X_val_numerical.columns = X_val_numerical.columns.astype(str)
    X_test_numerical.columns = X_test_numerical.columns.astype(str)

    # Scale
    scaler = StandardScaler()
    train_scaled_values = scaler.fit_transform(X_train_numerical)
    val_scaled_values = scaler.transform(X_val_numerical)
    test_scaled_values = scaler.transform(X_test_numerical)

    # Rebuild scaled DataFrames for train, val, test
    data_train = pd.DataFrame(train_scaled_values, columns=X_train_numerical.columns)
    data_train["index"] = data_train.index  # use current DataFrame index as conversation ID

    data_val = pd.DataFrame(val_scaled_values, columns=X_val_numerical.columns)
    data_val["index"] = data_val.index

    data_test = pd.DataFrame(test_scaled_values, columns=X_test_numerical.columns)
    data_test["index"] = data_test.index


    print("data shape = {0}\ndata_train shape = {1}\ndata_val shape = {2}\ndata_test shape = {3}".format(
        data.shape, data_train.shape, data_val.shape, data_test.shape))
    range_n_clusters = range(config["n_clusters_min"], config["n_clusters_max"])  # Desirable range

    # Segment parameters
    segment_window_length = 20  # Adjust as needed
    segment_stride = 20

    # Slice conversations into smaller time patches per split
    segments_train, origins_train = slice_timeseries_rowwise(data_train, segment_window_length, segment_stride)
    segments_val, origins_val = slice_timeseries_rowwise(data_val, segment_window_length, segment_stride)
    segments_test, origins_test = slice_timeseries_rowwise(data_test, segment_window_length, segment_stride)

    # Get labels for each segment
    labels_segments_train = labels_train[origins_train].reset_index(drop=True)
    labels_segments_val = labels_val[origins_val].reset_index(drop=True)
    labels_segments_test = labels_test[origins_test].reset_index(drop=True)

    print(data_train.columns)
    print(f"Segments_train.shape ->   {segments_train.shape}")

# ---- CAN"T USE cURRENTLY, BECAUSE segments trian might have adata from val too?
 #   scaler = StandardScaler()
#    segments_train = scaler.fit_transform(segments_train)
 #   segments_val = scaler.transform(segments_val)
 #   segments_test = scaler.transform(segments_test)

    # Train the tokenizer
    kmeans_model, knn_model, tokens_train, n_clusters = train_tokenizer(segments_train, range_n_clusters, knn_neighbors=config["knn_neighbors"])
    tokens_val = kmeans_model.predict(segments_val)
    tokens_test = kmeans_model.predict(segments_test)

    print(tokens_train)
    print(f"tokens_train.shape -> {tokens_train.shape}")

  #  plot_tsnePCAUMAP(TSNE, np.array(segments_train), labels_segments_train, config["perplexity"], config["random_state"], "of data_train", "no")
  #  plot_tsnePCAUMAP(TSNE, segments_train, kmeans_model.fit_predict(segments_train), config["perplexity"], config["random_state"], f"with {n_clusters} Clusters", "no")

    window_size = config["window_size"]  # Length of each sequence
    stride = config["stride"]  # Step size to slide the window (1 ensures maximum overlap)

    def group_tokens_by_conversation(tokens, origins):
        conv_token_map = defaultdict(list)
        for token, conv_id in zip(tokens, origins):
            conv_token_map[conv_id].append(token)
        return conv_token_map

    # Group tokens by conversation index
    train_token_dict = group_tokens_by_conversation(tokens_train, origins_train)
    val_token_dict = group_tokens_by_conversation(tokens_val, origins_val)
    test_token_dict = group_tokens_by_conversation(tokens_test, origins_test)

    # Convert to sorted list of sequences
    train_token_sequences = [train_token_dict[i] for i in sorted(train_token_dict.keys())]
    val_token_sequences = [val_token_dict[i] for i in sorted(val_token_dict.keys())]
    test_token_sequences = [test_token_dict[i] for i in sorted(test_token_dict.keys())]

    print(f"Number of sequences: {len(train_token_sequences)}")
    print(f"Length of first sequence: {len(train_token_sequences[0])}")
    print(f"First 10 tokens of first sequence: {train_token_sequences[0][:10]}")


    # Parameters
    vocab_size = n_clusters  # Set vocabulary size based on your tokens
    embedding_dim = config["embedding_dim"]
    window_size_skipgram = config["window_size_skipgram"]
    epochs = config["epochs"]


    def nce_loss(y_true, y_pred):
        return tf.nn.nce_loss(
            weights=tf.transpose(y_pred),
            biases=tf.zeros([y_pred.shape[1]]),
            labels=tf.reshape(y_true, (-1, 1)),
            inputs=y_pred,
            num_sampled=5,  # Negative samples
            num_classes=y_pred.shape[1]
        )
#    loss = nce_loss # NEEDs to be IMPLEMENTED FROM SCRATCH
    loss = config['skipgram_loss']

    # Train skip-gram model
    skipgram_model = train_skipgram(train_token_sequences, vocab_size, embedding_dim, window_size_skipgram, epochs, loss=loss)
    print("Skip-gram model trained!")

    # Step 2: Compute sequence embeddings
 #   train_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in segments_train])
 #   val_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in segments_train])
 #   test_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in segments_train])

    train_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in train_token_sequences])
    val_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in val_token_sequences])
    test_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in test_token_sequences])

    print(train_embeddings.shape); print(val_embeddings.shape); print(test_embeddings.shape)
    print(type(train_embeddings))

    labels_train_seq = np.array([labels_train[i] for i in sorted(train_token_dict.keys())])
    labels_val_seq = np.array([labels_val[i] for i in sorted(val_token_dict.keys())])
    labels_test_seq = np.array([labels_test[i] for i in sorted(test_token_dict.keys())])

    print(labels_train_seq)


  #  train_timeseries_embeddings = convertBackIntoTokenEmbeddings(train_token_sequences, segments_train, train_embeddings)
  #  val_timeseries_embeddings = convertBackIntoTokenEmbeddings(val_token_sequences, segments_train, val_embeddings)
  #  test_timeseries_embeddings = convertBackIntoTokenEmbeddings(test_token_sequences, segments_train, test_embeddings)
    trainValTest_embeddings = np.vstack([train_embeddings, val_embeddings, test_embeddings])
    labels_all = np.concatenate([labels_train_seq, labels_val_seq, labels_test_seq])

    indices_all = np.vstack([indices_train.reshape(-1, 1), indices_val.reshape(-1, 1), indices_test.reshape(-1, 1)])

    name_kwargs = {
        "nCl": n_clusters,
        "nN": config['knn_neighbors'],
        "winSize": window_size,
        "stride": stride,
        "winSizeSkip": window_size_skipgram,
        "nEmbeddings": embedding_dim
    }
    subfoldername = config["output_folder"]
    formatted_datetime = returnFormattedDateTimeNow()

    SaveEmbeddingsToOutput(trainValTest_embeddings, labels_all, subfoldername, formatted_datetime, indices_all, **name_kwargs)
 #   SaveEmbeddingsToOutput(trainValTest_embeddings, labels, subfoldername, formatted_datetime, indices_all, **name_kwargs)

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
    SaveEmbeddingsToOutput(train_embeddings, labels_train, subfoldername, formatted_datetime, indices_train, **name_kwargs_train)
    SaveEmbeddingsToOutput(val_embeddings, labels_val, subfoldername, formatted_datetime, indices_val, **name_kwargs_val)
    SaveEmbeddingsToOutput(test_embeddings, labels_test, subfoldername, formatted_datetime, indices_test, **name_kwargs_test)
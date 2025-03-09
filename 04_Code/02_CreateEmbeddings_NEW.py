import os
#import time
from datetime import datetime
#from fileinput import filename
#from os.path import abspath
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing.sequence import skipgrams
from utils00 import returnFilepathToSubfolder, doTrainValTestSplit, makeLabelsInt, doTrainValTestSplit222222, \
    readCsvAsDataframe


def find_optimal_clusters(data, range_n_clusters):
    """Find the optimal number of clusters using silhouette score."""
    best_n_clusters, best_score, best_model  = None, -1, None

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
    knn.fit(data, tokens)

    return kmeans, knn, tokens, n_clusters

def create_sequences(token_sequence, window_size, stride):
#    print(f"Number of overlapping sequences: {len(sequences)}");print(sequences)
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
    model.fit(pairs_context, pairs_target, epochs=epochs, batch_size=256, verbose=1)

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

    df = pd.DataFrame(embeddings)
    # Check if indices are available and add them as a column
    df_indices = pd.DataFrame({'Indices' : []})
    if indices is not None:
        df_indices = pd.DataFrame(indices)

    case_type = "Pitt" if "Pitt" in filepath_data else "Lu"
    case_type_str = f"_{case_type}_{setType}_" if setType != "NO" else f"_{case_type}_"

    filename_variables = "".join(f"{key}{value}".replace("{", "").replace("}", "") + "_" for key, value in kwargs.items()).rstrip("_")

    filename = "Embeddings" + case_type_str + filename_variables + "_" + formatted_datetime + ".csv"
    filenameFull = returnFilepathToSubfolder(filename, subfolderName)

    # Writing to CSV with pandas (which is generally faster)
    df.to_csv(filenameFull, index=False, header=False)

    df_labels = pd.DataFrame(labels)
    filename = "Labels" + case_type_str + filename_variables + "_" + formatted_datetime + ".csv"
    filenameFull = returnFilepathToSubfolder(filename, subfolderName)
    df_labels.to_csv(filenameFull, index=False, header=False)

    filename = "Indices" + case_type_str + filename_variables + "_" + formatted_datetime + ".csv"
    filenameFull = returnFilepathToSubfolder(filename, subfolderName)
    df_indices.to_csv(filenameFull, index=False, header=False)

    return

# Example Usage
if __name__ == "__main__":
#    abspath = "/home/vang/Downloads/"
    abspath = ""
    abspath = os.getcwd()
    timeSeriesDataPath = "/01_TimeSeriesData/"
    folderPath = abspath + timeSeriesDataPath
    filepath_data = "Lu_sR50_2025-01-06_01-40-21_output (Copy).csv"
    filepath_data = "Pitt_sR11025.0_2025-01-20_23-11-13_output.csv"
    filepath_data = "Pitt_output_sR11025_frameL2048_hopL512_thresh0.02_2025-02-22_14-49-06.csv"
    data = readCsvAsDataframe(folderPath, filepath_data)

    filepath_labels = "Lu_sR50_2025-01-06_01-40-21_output.csv"
    filepath_labels = "Pitt_sR11025.0_2025-01-20_23-12-07_labels.csv"
    filepath_labels = "Pitt_labels_sR11025_frameL2048_hopL512_thresh0.02_2025-02-22_14-49-06.csv"
    initial_labels = readCsvAsDataframe(folderPath, filepath_labels, dataFilename = "labels", as_series=True)

    print(f"----- BEFORE DROPPING NA -----")
    print(f"Labels shape is = {initial_labels.shape}")
    print(f"Data shape is = {data.shape}")
    # Drop NaN rows from data, # Reset indices after dropping rows
    data = data.dropna().reset_index(drop=True)
    # Ensure labels align with the updated data
    labels = initial_labels[data.index]
    labels = labels.reset_index(drop=True)

    labels = makeLabelsInt(labels)
    print(f"----- AFTER DROPPING NA -----")
    print(f"Labels shape is = {labels.shape}")
    print(f"Data shape is = {data.shape}\n")

    data_train, data_val, data_test, labels_train, labels_val, labels_test, val_ratio, indices_train, indices_val, indices_test = doTrainValTestSplit(data, labels)
    X_data = np.array(data); Y_targets = np.array(labels)
    print(f'\nLength of X is = {len(X_data)}. Length of Y is = {len(Y_targets)}')

    print(f"data shape = {data.shape}")
    print(f"data_train shape = {data_train.shape}")
    print(f"data_val shape = {data_val.shape}")
    print(f"data_test shape = {data_test.shape}")
    n_clusters_min = 5 # Was initially 2
    n_clusters_max = 30 # Was initially 10
    # Define the range for the number of clusters
    range_n_clusters = range(n_clusters_min, n_clusters_max)  # Desirable range

    data_train_np = np.array(data_train)

    def testTsne():
        # Apply t-SNE (reduce to 2D)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        data_train_tsne = tsne.fit_transform(data_train_np)
        # Scatter plot of t-SNE results
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data_train_tsne[:, 0], y=data_train_tsne[:, 1], hue=labels_train, palette="viridis", alpha=0.6)
        plt.title("t-SNE Visualization of data_train")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
      #  plt.show()

    # Train the tokenizer
    knn_neighbors = 50
    kmeans_model, knn_model, tokens_train, n_clusters = train_tokenizer(data_train, range_n_clusters, knn_neighbors = knn_neighbors)

    def testTsne2():
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(data_train)

        # Apply KMeans with 3 clusters
        #kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans_model.fit_predict(data_train)

        # Plot t-SNE with 3 clusters
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clusters, palette="Set1", alpha=0.7)
        plt.title(f"t-SNE Visualization with {n_clusters} Clusters")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
      #  plt.show()

    # Tokenize Train, Val, Test
    train_token_sequence = tokens_train.tolist()
    val_token_sequence = kmeans_model.predict(data_val).tolist()
    test_token_sequence = kmeans_model.predict(data_test).tolist()

    window_size = 10  # Length of each sequence
    stride = 1  # Step size to slide the window (1 ensures maximum overlap)

    train_sequences = create_sequences(train_token_sequence, window_size, stride)
    val_sequences = create_sequences(val_token_sequence, window_size, stride)
    test_sequences = create_sequences(test_token_sequence, window_size, stride)

    # Parameters
    vocab_size = n_clusters_max  # Set vocabulary size based on your tokens
    embedding_dim = 300
    window_size_skipgram = 20
    epochs = 1
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
    subfoldername = "02_Embeddings"
    SaveEmbeddingsToOutput(trainValTest_embeddings, labels, subfoldername, indices_all, **name_kwargs)

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

    # debugging
    for i in range(10):  # Έλεγχος σε τυχαία 10 δείγματα
        print(f"Index: {indices_all[i][0]}, Label: {labels[i]}, Embedding First 3 Values: {trainValTest_embeddings[i][:3]}")
    sorted_indices = np.argsort(indices_all[:, 0])
    trainValTest_embeddings = trainValTest_embeddings[sorted_indices]
    labels = labels[sorted_indices]

    print(labels)
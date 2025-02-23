import os
import time
from datetime import datetime
from fileinput import filename
from os.path import abspath
from utils00 import returnFilepathToSubfolder, doTrainValTestSplit, makeLabelsInt

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


def find_optimal_clusters(data, range_n_clusters):
    """Find the optimal number of clusters using silhouette score."""
    best_n_clusters = None
    best_score = -1
    best_model = None

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        print(f"Clusters = {n_clusters} -> Silhouette score ={score}")
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_model = kmeans

    return best_n_clusters, best_model


def train_tokenizer(data, range_n_clusters, knn_neighbors=5):
    """Train a tokenizer using K-means and k-NN."""
    # Step 1: Find optimal number of clusters
    n_clusters, kmeans = find_optimal_clusters(data, range_n_clusters)
    print(f"\nOptimal number of clusters: {n_clusters}")

    # Step 2: Assign tokens to data points
    tokens = kmeans.predict(data)
    print(f"Tokens assigned to first 5 data points: {tokens[:5]}")  # Print first 5 token assignments

    # Step 3: Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
    knn.fit(data, tokens)

    return kmeans, knn, tokens, n_clusters

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

def create_sequences(token_sequence, window_size, stride):
#    print(f"Number of overlapping sequences: {len(sequences)}")
    # Print the sequences
#    print(sequences)
    return [token_sequence[i:i + window_size] for i in range(0, len(token_sequence) - window_size + 1, stride)]



# ----- -----     SKIP GRAM     ----- -----
def create_skipgram_pairs(sequence, vocab_size, window_size=2, negative_samples=5):
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
    """
    Build a skip-gram model with a single hidden layer.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1),
        Flatten(),
        Dense(vocab_size, activation='softmax')  # Output layer for token prediction
    ])
    #model.compile(optimizer='adam', loss=loss)
    model.compile(optimizer='adam', loss=loss)
    return model

def train_skipgram(corpus, vocab_size, embedding_dim=50, window_size=2, epochs=10, loss= "sparse_categorical_crossentropy"):
    """
    Train a skip-gram model on the tokenized corpus.
    """
    # Flatten the corpus for skip-gram generation
    flat_corpus = [token for sequence in corpus for token in sequence]

    # Generate skip-gram pairs
    pairs, labels = create_skipgram_pairs(flat_corpus, vocab_size, window_size)

    # Build the skip-gram model
    model = build_skipgram_model(vocab_size, embedding_dim, loss)

    # Train the model
    pairs_context, pairs_target = pairs[:, 0], pairs[:, 1]
    model.fit(pairs_context, pairs_target, epochs=epochs, batch_size=256, verbose=1)

    return model

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
    # Show the result
    print(f"Shape of sequence embedding: {token_embeddings.shape}")
    print(f"Sequence embedding:\n{token_embeddings}")
    return token_embeddings

def SaveEmbeddingsToOutput(embeddings, labels, subfolderName, setType="NO", **kwargs):
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    df = pd.DataFrame(embeddings)
    caseType = "Pitt"
    if "Pitt" in filepath_data:
        caseType = "Pitt"
    if "Lu" in filepath_data:
        caseType = "Lu"
    if setType!= "NO":
        caseTypeString = f"_{caseType}_{setType}_"
    else:
        caseTypeString = f"_{caseType}_"

    filename_variables = "".join(f"{key}{value}".replace("{", "").replace("}", "") + "_" for key, value in name_kwargs.items()).rstrip("_")

    filename = "Embeddings" + caseTypeString + filename_variables + "_" + formatted_datetime + ".csv"
    filenameFull = returnFilepathToSubfolder(filename, subfolderName)

    # Writing to CSV with pandas (which is generally faster)
    df.to_csv(filenameFull, index=False, header=False)

    df_labels = pd.DataFrame(labels)
    filename = "Labels" + caseTypeString + formatted_datetime + ".csv"
    filenameFull = returnFilepathToSubfolder(filename, subfolderName)
    df_labels.to_csv(filenameFull, index=False, header=False)
    pass;


# Example Usage
if __name__ == "__main__":
#    abspath = "/home/vang/Downloads/"
    abspath = ""
    abspath = os.getcwd()
    timeSeriesDataPath = "/01_TimeSeriesData/"
    folderPath = abspath + timeSeriesDataPath
    filepath_data = "Lu_sR50_2025-01-06_01-40-21_output (Copy).csv"
    filepath_data = "Pitt_sR11025.0_2025-01-20_23-11-13_output.csv"
    data = returnData(folderPath, filepath_data)

    filepath_labels = "Lu_sR50_2025-01-06_01-40-21_output.csv"
    filepath_labels = "Pitt_sR11025.0_2025-01-20_23-12-07_labels.csv"
    initial_labels = returnLabels(folderPath, filepath_labels)

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

    data_train, data_val, data_test, labels_train, labels_val, labels_test, val_ratio = doTrainValTestSplit(
        data, labels)

    n_clusters_min = 5 # Was initially 2
    n_clusters_max = 30 # Was initially 10
    # Define the range for the number of clusters
    range_n_clusters = range(n_clusters_min, n_clusters_max)  # Desirable range

    # Train the tokenizer
    knn_neighbors = 50
    kmeans_model, knn_model, tokens_train, n_clusters = train_tokenizer(data_train, range_n_clusters, knn_neighbors = knn_neighbors)

    # Tokenize Train, Val, Test
    train_token_sequence = tokens_train.tolist()
    val_token_sequence = kmeans_model.predict(data_val).tolist() #knn_model.predict(data_val).tolist()
    test_token_sequence = kmeans_model.predict(data_test).tolist() #knn_model.predict(data_test).tolist()

    window_size = 10  # Length of each sequence
    stride = 1  # Step size to slide the window (1 ensures maximum overlap)

    train_sequences = create_sequences(train_token_sequence, window_size, stride)
    val_sequences = create_sequences(val_token_sequence, window_size, stride)
    test_sequences = create_sequences(test_token_sequence, window_size, stride)

#    tokenized_data = sequences
    """    # example of how tokenized_data should look like
        tokenized_data = [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [1, 3, 4, 6, 7]
        ]    
    """

    # Parameters
    vocab_size = n_clusters_max  # Set vocabulary size based on your tokens
    embedding_dim = 300
    window_size_skipgram = 20
    epochs = 10
    # loss = "nce" # NEEDs to be IMPLEMENTED FROM SCRATCH

    # Train skip-gram model
    skipgram_model = train_skipgram(train_sequences, vocab_size, embedding_dim, window_size_skipgram, epochs)  # , loss)
    print("Skip-gram model trained!")

    # Step 2: Compute sequence embeddings
    train_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in train_sequences])
    val_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in val_sequences])
    test_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in test_sequences])
    print(train_embeddings.shape)
    print(val_embeddings.shape)
    print(test_embeddings.shape)
    print(type(train_embeddings))

    train_timeseries_embeddings = convertBackIntoTokenEmbeddings(train_token_sequence, train_sequences, train_embeddings)
    val_timeseries_embeddings = convertBackIntoTokenEmbeddings(val_token_sequence, val_sequences, val_embeddings)
    test_timeseries_embeddings = convertBackIntoTokenEmbeddings(test_token_sequence, test_sequences, test_embeddings)
    trainValTest_embeddings = np.vstack([train_timeseries_embeddings, val_timeseries_embeddings, test_timeseries_embeddings])

    name_kwargs = {
        "nCl": n_clusters,
        "nN": knn_neighbors,
        "winSize": window_size,
        "stride": stride,
        "winSizeSkip": window_size_skipgram,
        "nEmbeddings": embedding_dim
    }
    subfoldername = "02_Embeddings"
    SaveEmbeddingsToOutput(trainValTest_embeddings, labels, subfoldername, **name_kwargs)
 #   SaveEmbeddingsToOutput(train_timeseries_embeddings, labels_train, "train", subfoldername, **name_kwargs)
 #   SaveEmbeddingsToOutput(val_timeseries_embeddings, labels_val,"val", subfoldername, **name_kwargs)
 #   SaveEmbeddingsToOutput(test_timeseries_embeddings, labels_test,"test", subfoldername, **name_kwargs)

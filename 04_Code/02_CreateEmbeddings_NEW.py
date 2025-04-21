import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from keras import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing.sequence import skipgrams

from utils00 import (
    makeLabelsInt,
    doTrainValTestSplit,
    readCsvAsDataframe, returnFormattedDateTimeNow, returnDataAndLabelsWithoutNA,
    returnDistribution, dropInstancesUntilClassesBalance, read_padded_csv_with_lengths, return_scaler_type,
)
from utils_Plots import (
    plotSilhouetteVsNClusters,
    plot_tsnePCAUMAP,
    plot_token_distribution_Bar,
    plot_token_waveforms,
    plot_token_spectrograms,
    plot_umap_of_segments,
    plot_tsne_of_segments
)


def find_optimal_clusters(data, n_clusters_list):
    """Find the optimal number of clusters using silhouette score."""
    best_n_clusters, best_score, best_model  = None, -1, None

    all_Silhouettes = []
    all_KMeans_times = []
    for n_clusters in n_clusters_list:
        start_time = time.perf_counter()  # Get current time at start

        kmeans = KMeans(n_clusters=n_clusters, n_init=config["n_init"], random_state=config["random_state"])
        try:
            cluster_labels = kmeans.fit_predict(data)
            score = silhouette_score(data, cluster_labels)
        except:
            score = -0.001001
        all_Silhouettes.append(score)

        end_time = time.perf_counter()  # Get current time at end
        kMeans_time = end_time - start_time  # Subtract the time at start and time at end, to find the total run time
        all_KMeans_times.append(kMeans_time)
        print(f"Clusters = {n_clusters} --- Training_Time = {kMeans_time:.6f} --- Silhouette score = {score:.6f}")
        if score > best_score:
            best_n_clusters, best_score, best_model = n_clusters, score, kmeans



    return best_n_clusters, best_model, best_score, all_Silhouettes, all_KMeans_times

def train_tokenizer(data, range_n_clusters, knn_neighbors=5):
    """Train a tokenizer using K-means and k-NN."""

    n_clusters, kmeans, best_silhouette, all_Silhouettes, all_KMeans_times = find_optimal_clusters(data, range_n_clusters)
    print(f"\nOptimal number of clusters: {n_clusters}")

    tokens = kmeans.predict(data)
    print(f"Tokens assigned to first 5 data points: {tokens[:5]}")  # Print first 5 token assignments

    # Step 3: Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors).fit(data, tokens)
    return kmeans, knn, tokens, n_clusters, best_silhouette, all_Silhouettes, all_KMeans_times

# ----- -----     SKIP GRAM     ----- -----
def generate_skipgram_pairs(sequence, vocab_size, window_size=2):
    """
    Generate skip-gram pairs using TensorFlow's skipgrams utility.
    """
    pairs, labels = skipgrams(
        sequence,
        vocabulary_size=vocab_size,
        window_size=window_size,
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

def train_skipgram_withinSameConversations(corpus, vocab_size, embedding_dim=50, window_size=2, epochs=10, loss="sparse_categorical_crossentropy"):
    """Train a skip-gram model on the tokenized corpus, respecting conversation boundaries."""

    all_pairs = []
    all_labels = []

    print(f"🧠 Training skip-gram across {len(corpus)} sequences")

    #  ✅ Respect conversation boundaries
    for idx, sequence in enumerate(corpus):
        if len(sequence) < 2:
            continue  # skip very short sequences

        # Generate skip-gram pairs for this sequence only
        pairs, labels = generate_skipgram_pairs(sequence, vocab_size, window_size)
        all_pairs.extend(pairs)
        all_labels.extend(labels)

        if idx < 3:  # Print a few examples
            print(f"\nSequence {idx} → {sequence[:10]}")
            print(f"Sample pairs: {pairs[:5]}")
            print(f"Sample labels: {labels[:5]}")

    # Convert to arrays
    all_pairs = np.array(all_pairs, dtype=np.int32)
    all_labels = np.array(all_labels, dtype=np.int32)

    # Split context and target
    pairs_context, pairs_target = all_pairs[:, 0], all_pairs[:, 1]

    print(f"\nTotal skip-gram pairs: {len(pairs_context)}")
    print(f"📏 Context shape: {pairs_context.shape}")
    print(f"🎯 Target shape: {pairs_target.shape}")

    # Build and train the model
    model = build_skipgram_model(vocab_size, embedding_dim, loss)
    history = model.fit(pairs_context, pairs_target, epochs=epochs, batch_size=config["batch_size"], verbose=1)

    return model, history

class SkipGramNCE(Model):
    def __init__(self, vocab_size, embedding_dim, num_negative_samples):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_negative_samples = num_negative_samples

        # "Hidden layer": learns the embeddings
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1, name="embedding_layer")

        # These are the output weights (used in the dot product for prediction)
        self.nce_weights = self.add_weight(
            shape=(vocab_size, embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="nce_weights"
        )
        self.nce_biases = self.add_weight(
            shape=(vocab_size,),
            initializer="zeros",
            trainable=True,
            name="nce_biases"
        )

    def call(self, inputs):
        # Forward pass returns the embedding
        return self.embedding(inputs)

    def compute_nce_loss(self, input_tokens, context_tokens):
        """
        input_tokens: center words (target tokens)
        context_tokens: context words (labels)
        """
        # Lookup embeddings for the input tokens
        input_embeds = self.embedding(input_tokens)  # shape = (batch_size, embedding_dim)

        # Compute NCE loss
        loss = tf.reduce_mean(tf.nn.nce_loss(
            weights=self.nce_weights,
            biases=self.nce_biases,
            labels=tf.reshape(context_tokens, [-1, 1]),
            inputs=input_embeds,
            num_sampled=self.num_negative_samples,
            num_classes=self.vocab_size
        ))
        return loss
def train_skipgram_with_nce(corpus, vocab_size, embedding_dim=300, window_size=6, epochs=10, num_negative_samples=5, batch_size=128):
    all_pairs = []
    # ✅ Respect conversation boundaries
    for sequence in corpus:
        if len(sequence) < 2:
            continue
        pairs, _ = skipgrams(sequence, vocabulary_size=vocab_size, window_size=window_size, negative_samples=0)
        all_pairs.extend(pairs)

    all_pairs = tf.convert_to_tensor(all_pairs, dtype=tf.int32)
    input_tokens = all_pairs[:, 0]
    context_tokens = all_pairs[:, 1]

    dataset = tf.data.Dataset.from_tensor_slices((input_tokens, context_tokens))
    dataset = dataset.shuffle(100000).batch(batch_size)

    model = SkipGramNCE(vocab_size, embedding_dim, num_negative_samples)

    opt_name = config['optimizer_skipgram'].lower() # Adagrad 0.001 To match the paper
    if opt_name == 'adam':
        optimizer = tf.keras.optimizers.Adam()
    elif opt_name == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001)
    elif opt_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD()
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    history = {"loss": []}
    for epoch in range(epochs):
        start_time = time.time()

        total_loss = 0
        steps = 0
        for batch_inputs, batch_context in dataset:
            with tf.GradientTape() as tape:
                loss = model.compute_nce_loss(batch_inputs, batch_context)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += loss.numpy()
            steps += 1

        end_time = time.time(); epoch_time = end_time - start_time
        avg_loss = total_loss / steps
        history["loss"].append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f} sec")

    return model, history

# Function to get embeddings for each sequence directly
def get_sequence_embedding(token_sequence, model):
    """Compute sequence embedding by averaging token embeddings."""
    if config['skipgram_loss'] == "NCE":
        token_embedding = np.array([model.embedding.get_weights()[0][token] for token in token_sequence])
    else:
        token_embedding = np.array([model.layers[0].get_weights()[0][token] for token in token_sequence])

    # Signal2Vec uses Average, but mentions weighted as a potential future study
    if config["sequenceEmbeddingAveragingMethod"] == "Average":
        sequence_embedding = np.mean(token_embedding, axis=0)
    elif config["sequenceEmbeddingAveragingMethod"] == "Weighted":
        weights = np.arange(1, len(token_sequence) + 1)  # Linear weight increase
        sequence_embedding = np.average(token_embedding, axis=0, weights=weights)
    else:
        raise "NO sequenceEmbeddingAveragingMethod DEFINED"

    return sequence_embedding

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

def scale_split_data(scaler_class, data, indices, enable_scaling=False, fit=False, rowWiseScaling=True):
    subset = data.iloc[indices].copy()
    if not enable_scaling:
        return subset.reset_index(drop=True)

    if rowWiseScaling:
        def scale_row(row):
            values = row.dropna().values.astype(float).reshape(-1, 1)
            scaled = scaler_class().fit_transform(values).flatten()
            return pd.Series(scaled)

        scaled_df = subset.apply(scale_row, axis=1)
    else:
        X = subset.drop(columns=["index"]).copy()
        X.columns = X.columns.astype(str)

        if fit:
            scaler = scaler_class()
            scaled_values = scaler.fit_transform(X)
        else:
            scaled_values = scaler_class().transform(X)

        scaled_df = pd.DataFrame(scaled_values, columns=X.columns)
        scaled_df["index"] = scaled_df.index
    return scaled_df.reset_index(drop=True)

def group_tokens_by_conversation(tokens, origins):
    conv_token_map = defaultdict(list)
    for token, conv_id in zip(tokens, origins):
        conv_token_map[conv_id].append(token)
    return conv_token_map


def saveResultsFile(filepath_data, filepath_labels, all_Silhouettes, all_KMeans_times, n_clusters_list, allDataShapes, allSegmenthapes,
                    skipgram_history, total_skipgram_time, tokens_train, subfolderName, formatted_datetime, setType="NO", **kwargs):
    case_type = "Pitt" if "Pitt" in filepath_data else "Lu"
    case_type_str = f"{case_type}_{setType}" if setType != "NO" else f"_{case_type}_"

    filename_variables = "".join(
        f"{key}{value}".replace("{", "").replace("}", "") + "_" for key, value in kwargs.items()).rstrip("_")

    # Helper function to generate paths dynamically
    def generate_path(prefix):
        return f"{subfolderName}/{prefix}_{case_type_str}{filename_variables}_{formatted_datetime}.csv"

    if isinstance(skipgram_history, dict):
        df_history = pd.DataFrame(skipgram_history)
    else:
        df_history = pd.DataFrame(skipgram_history.history)
    df_history.insert(0, "Epoch", range(1, len(df_history) + 1))  # Add epoch numbers
    df_history = df_history.round(6)

    filenameFull = generate_path("Results")
    # Save everything into a single CSV file
    with open(filenameFull, "w") as f:
        f.write("INPUT FILENAMES:\n")
        f.write(filepath_data);f.write("\n")
        f.write(filepath_labels)

        f.write("\nallDataShapes:")
        json.dump(allDataShapes, f, indent=4)

        f.write("\nallSegmenthapes:")
        json.dump(allSegmenthapes, f, indent=4)

        f.write("\nClusters --- Training Time --- Silhouette Scores:\n")
        for i in range(0, len(all_Silhouettes)):
            f.write(f"{n_clusters_list[i]} --- {all_KMeans_times[i]:.6f}--- {all_Silhouettes[i]:.6f} \n")

        config_serializable = config.copy()
        config_serializable["optimizer_skipgram"] = str(config["optimizer_skipgram"])  # Convert optimizer to string
        config_serializable["skipgram_loss"] = str(config["skipgram_loss"])  # Convert L2
        config_serializable["scaler"] = str(config.get("scaler", ""))
        f.write("\nTHE WHOLE CONFIG FOLLLOWS:")
        json.dump(config_serializable, f, indent=4)

        f.write("\nSKIPGRAM MODEL HISTORY:\n")
        df_history.to_csv(f, index=False)

        f.write(f"Total Skipgram time: {total_skipgram_time:.2f} seconds\n")

        returnDistribution(tokens_train, "Token", file=f)

    return

def slice_timeseries_rowwise(data, lengths, window_length, stride):
    segments = []
    origins = []

    for idx, row in data.iterrows():
        row_values = row.values[:lengths[idx]]  # only use the real part
        for i in range(0, len(row_values) - window_length + 1, stride):
            segment = row_values[i:i + window_length]
            segments.append(segment)
            origins.append(idx)

    return np.array(segments), np.array(origins)


# Example Usage
def mainLogic():
    timeSeriesDataPath = "/01_TimeSeriesData/"
    folderPath = os.getcwd() + timeSeriesDataPath

#    data = readCsvAsDataframe(folderPath, filepath_data)
    data, lengths = read_padded_csv_with_lengths(os.path.join(folderPath, filepath_data))
    initial_labels = readCsvAsDataframe(folderPath, filepath_labels, dataFilename = "labels", as_series=True)

    data, labels = returnDataAndLabelsWithoutNA(data, initial_labels, addIndexColumn=True)
    labels = makeLabelsInt(labels)
    print_data_info(data, labels, "AFTER DROPPING NA")

    # ----- NAKE COUNT OF 0s AND 1s BE THE SAME -----
  #  data, labels = dropInstancesUntilClassesBalance(data, labels)

    _, _, _, _, _, _, val_ratio, indices_train, indices_val, indices_test = doTrainValTestSplit(data, labels)

    # Scale and fit on training
    data_train = scale_split_data(config["scaler"], data, indices_train, enable_scaling=config["enable_scaling"], fit=True, rowWiseScaling=config["rowWiseScaling"])
    data_val = scale_split_data(config["scaler"], data, indices_val, enable_scaling=config["enable_scaling"], rowWiseScaling=config["rowWiseScaling"])
    data_test = scale_split_data(config["scaler"], data, indices_test, enable_scaling=config["enable_scaling"], rowWiseScaling=config["rowWiseScaling"])

    labels_train = pd.Series(labels[indices_train]).reset_index(drop=True)
    labels_val = pd.Series(labels[indices_val]).reset_index(drop=True)
    labels_test = pd.Series(labels[indices_test]).reset_index(drop=True)

    lengths_train = [lengths[i] for i in indices_train]
    lengths_val = [lengths[i] for i in indices_val]
    lengths_test = [lengths[i] for i in indices_test]

    print("data shape = {0}\ndata_train shape = {1}\ndata_val shape = {2}\ndata_test shape = {3}".format(
        data.shape, data_train.shape, data_val.shape, data_test.shape))
    n_clusters_list = config["n_clusters_list"]

    # Segment parameters
    segment_window_length = config["window_size"]
    segment_stride = config["stride"]

    # Slice conversations into smaller time patches per split
  #  segments_train, origins_train = slice_timeseries_rowwise(data_train, segment_window_length, segment_stride)
 #   segments_val, origins_val = slice_timeseries_rowwise(data_val, segment_window_length, segment_stride)
  #  segments_test, origins_test = slice_timeseries_rowwise(data_test, segment_window_length, segment_stride)
    segments_train, origins_train = slice_timeseries_rowwise(data_train, lengths_train, segment_window_length, segment_stride)
    segments_val, origins_val = slice_timeseries_rowwise(data_val, lengths_val, segment_window_length, segment_stride)
    segments_test, origins_test = slice_timeseries_rowwise(data_test, lengths_test, segment_window_length, segment_stride)

    # Get labels for each segment
    labels_segments_train = labels_train[origins_train].reset_index(drop=True)
    labels_segments_val = labels_val[origins_val].reset_index(drop=True)
    labels_segments_test = labels_test[origins_test].reset_index(drop=True)

    print(data_train.columns)
    print(f"Segments_train.shape ->   {segments_train.shape}")
    print(f"Origins_train.shape ->   {origins_train.shape}")

    # Train the tokenizer
    kmeans_model, knn_model, tokens_train, n_clusters, best_silhouette, all_Silhouettes, all_KMeans_times = train_tokenizer(
        segments_train, n_clusters_list, knn_neighbors=config["knn_neighbors"])
    best_silhouette = np.round(best_silhouette, 4)

 #   plotSilhouetteVsNClusters(n_clusters_list, all_Silhouettes)

    # Token assignment step — we train a classifier (k-NN) on the clustered segments
    # as suggested in Signal2Vec: token extraction (KMeans), token assignment (classifier)
    tokens_train = knn_model.predict(segments_train)  # Optional: use classifier output
    tokens_val = knn_model.predict(segments_val)
    tokens_test = knn_model.predict(segments_test)

    print(tokens_train)
    print(f"tokens_train.shape -> {tokens_train.shape}")

  #  plot_tsnePCAUMAP(TSNE, np.array(segments_train), labels_segments_train, config["perplexity"], config["random_state"], "of data_train", "no")
  #  plot_tsnePCAUMAP(TSNE, segments_train, kmeans_model.fit_predict(segments_train), config["perplexity"], config["random_state"], f"with {n_clusters} Clusters", "no")

    window_size = config["window_size"]  # Length of each sequence
    stride = config["stride"]  # Step size to slide the window (1 ensures maximum overlap)

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
    print(f"First 10 tokens of first sequence: {train_token_sequences[0][:10]}\n")

    # Parameters
    vocab_size = n_clusters  # Set vocabulary size based on your tokens
    embedding_dim = config["embedding_dim"]
    window_size_skipgram = config["window_size_skipgram"]
    epochs = config["epochs"]
    loss = config['skipgram_loss']

    tokens, counts = returnDistribution(tokens_train)
    print('\n Distribution of most common tokens')
 #   plot_token_distribution_Bar(tokens_train)

    # Example: Visualize top 5 most common tokens
    print("\nVisualizing plots for most common tokens:")
 #   top_tokens = tokens[np.argsort(-counts)[:5]]  # Get top 5 by frequency
 #   for token_id in top_tokens:
 #       print("waveforms for most common tokens")
#        plot_token_waveforms(segments_train, tokens_train, token_id, sample_rate=11025, n_samples=5)
    #    print("\n🎨 Spectrograms for most common tokens:")
   #     plot_token_spectrograms(segments_train, tokens_train, token_id, sample_rate=11025, n_samples=5)

  #  print('\n TSNE for most common tokens')
 #   plot_tsne_of_segments(segments_train, tokens_train, perplexity=config["perplexity"], random_state=config["random_state"])
 #   print('\n UMAP for most common tokens')
 #   plot_umap_of_segments(segments_train, tokens_train)

    # Train skip-gram model
    start_time = time.time()
    if loss == "NCE":
        skipgram_model, skipgram_history = train_skipgram_with_nce(train_token_sequences, vocab_size, embedding_dim, window_size_skipgram, epochs)
    else:
    # skipgram_model = train_skipgram(train_token_sequences, vocab_size, embedding_dim, window_size_skipgram, epochs, loss=loss)
        skipgram_model, skipgram_history = train_skipgram_withinSameConversations(train_token_sequences, vocab_size, embedding_dim, window_size_skipgram, epochs, loss=loss)
    total_skipgram_time = time.time() - start_time
    print(f"Skip-gram model trained!\nTotal Skipgram time: {total_skipgram_time:.2f} seconds\n")

    train_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in train_token_sequences])
    val_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in val_token_sequences])
    test_embeddings = np.array([get_sequence_embedding(seq, skipgram_model) for seq in test_token_sequences])

    print(train_embeddings.shape); print(val_embeddings.shape); print(test_embeddings.shape)
    print(type(train_embeddings))

    labels_train_seq = np.array([labels_train[i] for i in sorted(train_token_dict.keys())])
    labels_val_seq = np.array([labels_val[i] for i in sorted(val_token_dict.keys())])
    labels_test_seq = np.array([labels_test[i] for i in sorted(test_token_dict.keys())])

    print(labels_train_seq)

    trainValTest_embeddings = np.vstack([train_embeddings, val_embeddings, test_embeddings])
    labels_all = np.concatenate([labels_train_seq, labels_val_seq, labels_test_seq])

    indices_all = np.vstack([indices_train.reshape(-1, 1), indices_val.reshape(-1, 1), indices_test.reshape(-1, 1)])

    scalerName = return_scaler_type(str(config.get("scaler", "")), config['enable_scaling'])
    name_kwargs = {
        "scaler": scalerName,
        "sil": best_silhouette,
        "nCl": n_clusters,
        "nN": config['knn_neighbors'],
        "wSize": window_size,
        "str": stride,
        "wSizeSG": window_size_skipgram,
        "SGLoss": loss,
        "nEmbd": embedding_dim
    }
    subfoldername = config["output_folder"]
    formatted_datetime = returnFormattedDateTimeNow()

    SaveEmbeddingsToOutput(trainValTest_embeddings, labels_all, subfoldername, formatted_datetime, indices_all, **name_kwargs)

    name_kwargs_train = { "train": "Set", **name_kwargs}
    name_kwargs_val = {"val": "Set", **name_kwargs}
    name_kwargs_test = {"test": "Set", **name_kwargs}
    SaveEmbeddingsToOutput(train_embeddings, labels_train, subfoldername, formatted_datetime, indices_train, **name_kwargs_train)
    SaveEmbeddingsToOutput(val_embeddings, labels_val, subfoldername, formatted_datetime, indices_val, **name_kwargs_val)
    SaveEmbeddingsToOutput(test_embeddings, labels_test, subfoldername, formatted_datetime, indices_test, **name_kwargs_test)

    allDataShapes = {
        "data": data.shape,
        "data_train": data_train.shape,
        "data_val": data_val.shape,
        "data_test": data_test.shape
    }
    allSegmenthapes = {
        "segments_train": segments_train.shape,
        "segments_val": segments_val.shape,
        "segments_test": segments_test.shape,
    }

    saveResultsFile(filepath_data, filepath_labels, all_Silhouettes, all_KMeans_times, n_clusters_list, allDataShapes,
                    allSegmenthapes, skipgram_history, total_skipgram_time, tokens_train, subfoldername, formatted_datetime, **name_kwargs)

 #   plot_umap_of_segments(train_embeddings, labels_train_seq)
    plotSilhouetteVsNClusters(n_clusters_list, all_Silhouettes)


filepath_data = "Lu_sR50_2025-01-06_01-40-21_output (Copy).csv"
filepath_data = "Pitt_sR11025.0_2025-01-20_23-11-13_output.csv"
filepath_data = "Pitt_output_sR11025_frameL2048_hopL512_thresh0.02_2025-02-22_14-49-06.csv"
filepath_data = "Pitt_output_sR11025_frameL2048_hopL512_thresh0.0_2025-03-25_17-18-31.csv"
filepath_data = "Pitt_output_sR11025_frameL2048_hopL512_thresh0.02_2025-03-26_01-39-56.csv"
filepath_data = "Pitt_output_sR11025_frameL2048_hopL512_thresh0.02_2025-03-27_01-16-49.csv"
filepath_data = "Pitt_output_sR11025_frameL2048_hopL512_thresh0.02_2025-03-27_22-39-38.csv"
filepath_data = "Pitt_output_sR11025_frameL2048_hopL512_thresh0.02_2025-04-01_22-39-34.csv"
filepath_data = "Pitt_output_sR1100_frameL2048_hopL512_thresh0.02_2025-04-01_22-54-29.csv"
filepath_data = "Pitt_output_sR300_frameL2048_hopL512_thresh0.02_2025-04-02_18-21-08.csv"
filepath_data = "Pitt_output_raw_sR300_frameL2048_hopL512_thresh0.02_2025-04-08_00-11-56.csv"
filepath_data = "Pitt_output_raw_sR100_frameL2048_2025-04-19_22-01-54.csv"
filepath_data = "Pitt_output_raw_sR300_frameL2048_2025-04-20_21-44-41.csv"

filepath_labels = "Lu_sR50_2025-01-06_01-40-21_output.csv"
filepath_labels = "Pitt_sR11025.0_2025-01-20_23-12-07_labels.csv"
filepath_labels = "Pitt_labels_sR11025_frameL2048_hopL512_thresh0.02_2025-02-22_14-49-06.csv"
filepath_labels = "Pitt_labels_sR11025_frameL2048_hopL512_thresh0.0_2025-03-25_17-18-31.csv"
filepath_labels = "Pitt_labels_sR11025_frameL2048_hopL512_thresh0.02_2025-03-26_01-39-56.csv"
filepath_labels = "Pitt_labels_sR11025_frameL2048_hopL512_thresh0.02_2025-03-27_01-16-49.csv"
filepath_labels = "Pitt_labels_sR11025_frameL2048_hopL512_thresh0.02_2025-03-27_22-39-38.csv"
filepath_labels = "Pitt_labels_sR11025_frameL2048_hopL512_thresh0.02_2025-04-01_22-39-34.csv"
filepath_labels = "Pitt_labels_sR1100_frameL2048_hopL512_thresh0.02_2025-04-01_22-54-29.csv"
filepath_labels = "Pitt_labels_sR300_frameL2048_hopL512_thresh0.02_2025-04-02_18-21-08.csv"
filepath_labels = "Pitt_labels_raw_sR300_frameL2048_hopL512_thresh0.02_2025-04-08_00-11-56.csv"
filepath_labels = "Pitt_labels_raw_sR100_frameL2048_2025-04-19_22-01-54.csv"
filepath_labels = "Pitt_labels_raw_sR300_frameL2048_2025-04-20_21-44-41.csv"

# Configuration dictionary to store hyperparameters and settings
config = {
    "n_clusters_min": 2,        # Min number of clusters for KMeans - 2
    "n_clusters_max": 10,       # Max number of clusters for KMeans - 10
  #  "n_clusters_list": range(config["n_clusters_min"], config["n_clusters_max"],
    "n_clusters_list": [150, 350, 500, 700, 1000, 1500], #[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 1500, 2000, 3000, 5000],
  #  "n_clusters_list": [10, 50],# 200, 250, 300, 350],
  #  "n_clusters_list": [2000, 3000, 5000, 10000],
    "knn_neighbors": 15,        # Number of neighbors for k-NN - 50
    "window_size": 256,    # 2       # Window size for sequence generation - 10
    "stride": 128,          # 8      # Stride for sequence generation - 1
    "embedding_dim": 150,       # Dimension of word embeddings - 300
    "window_size_skipgram": 6, # - 20
    "epochs": 10,                # Number of training epochs
    "optimizer_skipgram": 'adam', # Adagrad in nalmpantis paper
    "skipgram_loss": "NCE", # "sparse_categorical_crossentropy", "NCE
    "sequenceEmbeddingAveragingMethod": "Average", # "Average", "Weighted"
    "batch_size": 64,          # Batch size for training
    "perplexity": 30,           # t-SNE perplexity
    "random_state": 42,         # Random state for reproducibility
    "n_init": 50, #default 10 in kmeans. use 50 to help avoid collapse
    "output_folder": "02_Embeddings",  # Folder for saving embeddings
    "enable_scaling": True,
    "scaler": MinMaxScaler,  # MinMaxScaler(), StandardScaler(), "noScaling"
    "rowWiseScaling": True
}
# e.g. paper -> embedding dim = 300, window size = 6, etc

mainLogic()
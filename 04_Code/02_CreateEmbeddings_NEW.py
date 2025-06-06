import os
import time
import traceback
from collections import defaultdict
import numpy as np
import pandas as pd

from keras import Model
from keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing.sequence import skipgrams
from umap import UMAP

from utils00 import (
    makeLabelsInt,
    doTrainValTestSplit,
    readCsvAsDataframe, returnFormattedDateTimeNow, returnDataAndLabelsWithoutNA,
    returnDistribution, dropInstancesUntilClassesBalance, read_file_returnPadded_And_lengths, return_scaler_type,
    saveResultsFile02, SaveEmbeddingsToOutput02, print_data_info,
)
from utils_Plots import (
    plotSilhouetteVsNClusters,
    plot_tsnePCAUMAP,
    plot_token_distribution_Bar,
    plot_token_waveforms,
    plot_token_spectrograms,
    plot_umap_of_segments,
    plot_tsne_of_segments, plot_clustering_metrics, analyze_all_embedding_plots, plotSkipgramLossVsEpoch,
    compare_token_assignments
)


def find_optimal_clusters(data, n_param_list, clusteringAlgorithm="KMEANS", eps_list=None):
    best_param_combo, best_score, best_model = None, -1, None

    all_metrics = {
        "silhouette": [],
        "ch_index": [],
        "db_index": [],
        "times": [],
        "n_clusters": []
    }

    # Choose loop structure
    if clusteringAlgorithm == "KMEANS":
        param_grid = [(param,) for param in n_param_list]
    elif clusteringAlgorithm == "DBSCAN":
        eps_list = eps_list or [0.5]
        param_grid = [(eps, min_samples) for eps in eps_list for min_samples in n_param_list]
    else:
        raise ValueError(f"Unsupported clustering algorithm: {clusteringAlgorithm}")

    for param in param_grid:
        start_time = time.perf_counter()

        # Create model
        if clusteringAlgorithm == "KMEANS":
            n_clusters = param[0]
            model = KMeans(n_clusters=n_clusters, n_init=config["n_init"], random_state=config["random_state"])
        else:  # DBSCAN
            eps, min_samples = param
            model = DBSCAN(eps=eps, min_samples=min_samples)

        try:
            cluster_labels = model.fit_predict(data)
            n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            if n_clusters_found < 2:
                print(f"⚠️ Skipping {param} — only {n_clusters_found} cluster(s)")
                continue

            sil = silhouette_score(data, cluster_labels)
            ch = calinski_harabasz_score(data, cluster_labels)
            db = davies_bouldin_score(data, cluster_labels)
        except Exception as e:
            print(f"❌ Error for param={param}: {e}")
            traceback.print_exc()
            sil, ch, db = -9999, -9999, 9999
            model = None

        elapsed = time.perf_counter() - start_time

        all_metrics["silhouette"].append(sil)
        all_metrics["ch_index"].append(ch)
        all_metrics["db_index"].append(db)
        all_metrics["times"].append(elapsed)
        all_metrics["n_clusters"].append(param)

        param_str = f"n_clusters={param[0]}" if clusteringAlgorithm == "KMEANS" else f"eps={param[0]}, min_samples={param[1]}"
        print(f"method={clusteringAlgorithm} | {param_str} | Sil: {sil:.4f} | CH: {ch:.2f} | DBI: {db:.4f} | Time: {elapsed:.2f}s")

        if model is not None and sil > best_score:
            best_score = sil
            best_model = model
            best_param_combo = param

    return best_param_combo, best_model, best_score, all_metrics

def train_tokenizer(data, range_n_clusters, knn_neighbors=5, clusteringAlgorithm="KMEANS", eps_list=None):
    """Train a tokenizer using K-means and k-NN."""

    best_param_combo, best_model, best_silhouette, all_metrics = find_optimal_clusters(data, range_n_clusters, clusteringAlgorithm, eps_list)

    if clusteringAlgorithm == "KMEANS":
        n_clusters = best_param_combo[0]
    else:
        cluster_labels = best_model.labels_
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    print(f"\nOptimal number of clusters: {n_clusters}")

    if hasattr(best_model, "predict"):
        tokens_from_kmeans = best_model.predict(data)
        print(f"Tokens assigned to first 5 data points: {tokens_from_kmeans[:5]}")  # Print first 5 token assignments
    else:
        tokens_from_kmeans = best_model.labels_
        print(f"Tokens assigned to first 5 data points: {tokens_from_kmeans[:5]}")


    # Step 3: Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors).fit(data, tokens_from_kmeans)
    return best_model, knn, tokens_from_kmeans, n_clusters, best_silhouette, all_metrics

# ----- -----     SKIP GRAM     ----- -----
def generate_skipgram_pairs(sequence, vocab_size, window_size=2):
    """Generate skip-gram pairs using TensorFlow's skipgrams utility."""
    pairs, labels = skipgrams(
        sequence,
        vocabulary_size=vocab_size,
        window_size=window_size,
        negative_samples=0
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


    early_stopping_callback = EarlyStopping(
        monitor='loss',  # What to monitor ('loss', or 'val_loss' if validation split)
        patience=config["early_stopping_patience"],  # How many epochs without improvement to wait
        min_delta=config["early_stopping_min_delta"],  # Minimum change to be considered improvement
        restore_best_weights=True,  # Restore best model weights automatically
        verbose=1
    ) if config.get("early_stopping", False) else None
    callbacks = [early_stopping_callback] if early_stopping_callback else None

    history = model.fit(pairs_context, pairs_target, epochs=epochs, batch_size=config["batch_size"], verbose=1, callbacks=callbacks)

    epochEarlyStopped = early_stopping_callback.stopped_epoch if early_stopping_callback and early_stopping_callback.stopped_epoch > 0 else None

    return model, history, epochEarlyStopped

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
    if vocab_size <= num_negative_samples:
        num_negative_samples = vocab_size - 1
        print(f"⚠️ Reducing num_negative_samples to {num_negative_samples} due to small vocab_size = {vocab_size}")

    all_pairs = []
    # ✅ Respect conversation boundaries
    for sequence in corpus:
        if len(sequence) < 2:
            continue
        pairs, _ = skipgrams(sequence, vocabulary_size=vocab_size, window_size=window_size, negative_samples=0)
        all_pairs.extend(pairs)

    all_pairs = np.array(all_pairs, dtype=np.int32)
    if all_pairs.ndim != 2 or all_pairs.shape[1] != 2:
        raise ValueError(f"Invalid shape for skip-gram pairs: {all_pairs.shape}. Expected shape (N, 2).")
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

    # Early stopping config
    early_stopping = config.get("early_stopping", False)
    patience = config.get("early_stopping_patience", 3)
    min_delta = config.get("early_stopping_min_delta", 1e-4)

    best_loss = float('inf')
    epochs_without_improvement = 0
    best_weights = None
    epochEarlyStopped = None

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

        # early stopping check
        if early_stopping:
            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                best_weights = model.get_weights()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"✅ Early stopping triggered at epoch {epoch + 1}.")
                epochEarlyStopped = epoch + 1
                break

    # after all epochs
    if early_stopping and epochs_without_improvement >= patience and best_weights is not None:
        model.set_weights(best_weights)
        print(f"📦 Best model weights restored after early stopping. Patience = {patience}, and epochEarlyStopped = {epochEarlyStopped}")

    return model, history, epochEarlyStopped

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
        raise ValueError("NO sequenceEmbeddingAveragingMethod DEFINED")

    return sequence_embedding

def scale_split_data(scaler_class, data, indices, enable_scaling=False, fit=False, rowWiseScaling=True):
    subset = data.iloc[indices].copy()
    if not enable_scaling:
        return subset.reset_index(drop=True)

    if rowWiseScaling:
        def scale_row(row):
            # Drop non-signal columns like 'index' if present
            dropped_cols = []
            if "index" in row.index:
                dropped_cols.append("index")
                row = row.drop("index")
       #     if dropped_cols:
       #         print(f"⚠️ Dropping columns from row {row.name}: {dropped_cols}")

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
     #   scaled_df["index"] = scaled_df.index
        scaled_df["index"] = subset["index"].values
    return scaled_df.reset_index(drop=True)

def group_tokens_by_conversation(tokens, origins):
    conv_token_map = defaultdict(list)
    for token, conv_id in zip(tokens, origins):
        conv_token_map[conv_id].append(token)
    return conv_token_map

def slice_timeseries_rowwise(data, lengths, window_length, stride, pad_short=True):
    segments = []
    origins = []

    for idx, row in data.iterrows():
        row_values = row.values[:lengths[idx]]  # only use the real part

        if len(row_values) < window_length:
            if pad_short:
                padded = np.pad(row_values, (0, window_length - len(row_values)), mode='constant')
                segments.append(padded)
                origins.append(idx)
            else:
                print(f"⚠️ Skipping sample {idx} — too short (length={len(row_values)}, needed={window_length})")
            continue

        for i in range(0, len(row_values) - window_length + 1, stride):
            segment = row_values[i:i + window_length]
            segments.append(segment)
            origins.append(idx)

    print(f"\n✅ Finished slicing: {len(segments)} segments from {len(set(origins))} samples (skipped {len(data) - len(set(origins))})")

    return np.array(segments), np.array(origins)

# Example Usage
def mainLogic():
    timeSeriesDataPath = "/01_TimeSeriesData/"
    folderPath = os.getcwd() + timeSeriesDataPath

#    data = readCsvAsDataframe(folderPath, filepath_data)
    data, lengths = read_file_returnPadded_And_lengths(os.path.join(folderPath, filepath_data))
    initial_labels = readCsvAsDataframe(folderPath, filepath_labels, dataFilename = "labels", as_series=True)

    data, labels = returnDataAndLabelsWithoutNA(data, initial_labels, addIndexColumn=True)
    labels = makeLabelsInt(labels)
    print_data_info(data, labels, "AFTER DROPPING NA")

    # ----- NAKE COUNT OF 0s AND 1s BE THE SAME -----
  #  data, labels = dropInstancesUntilClassesBalance(data, labels)

    _, _, _, _, _, _, val_ratio, indices_train, indices_val, indices_test = doTrainValTestSplit(data, labels, random_state=config['random_state'])

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

    data_train = data_train.drop(columns=["index"], errors="ignore")
    data_val = data_val.drop(columns=["index"], errors="ignore")
    data_test = data_test.drop(columns=["index"], errors="ignore")

    # Slice conversations into smaller time patches per split
    segments_train, origins_train = slice_timeseries_rowwise(data_train, lengths_train, segment_window_length, segment_stride)
    segments_val, origins_val = slice_timeseries_rowwise(data_val, lengths_val, segment_window_length, segment_stride)
    segments_test, origins_test = slice_timeseries_rowwise(data_test, lengths_test, segment_window_length, segment_stride)

    # Get labels for each segment
    labels_segments_train = labels_train[origins_train].reset_index(drop=True)
    labels_segments_val = labels_val[origins_val].reset_index(drop=True)
    labels_segments_test = labels_test[origins_test].reset_index(drop=True)

    print(f"data_train.columns = {data_train.columns}")
    print(f"Segments_train.shape ->   {segments_train.shape}")
    print(f"Origins_train.shape ->   {origins_train.shape}")

    # Train the tokenizer
    kmeans_model, knn_model, tokens_from_kmeans, n_clusters, best_silhouette, all_metrics = train_tokenizer(
        segments_train, n_clusters_list, knn_neighbors=config["knn_neighbors"], clusteringAlgorithm=config['clusteringAlgorithm'], eps_list = config['dbscan_eps_list'])
    best_silhouette = np.round(best_silhouette, 4)

    # Token assignment step — we train a classifier (k-NN) on the clustered segments
    # as suggested in Signal2Vec: token extraction (KMeans), token assignment (classifier)
    tokens_train = tokens_from_kmeans
  #  tokens_train = knn_model.predict(segments_train)  # Optional: use classifier output
    tokens_val = knn_model.predict(segments_val)
    tokens_test = knn_model.predict(segments_test)

    print(tokens_train)
    print(f"tokens_train.shape -> {tokens_train.shape}")

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
    num_negative_samples = config['num_negative_samples']

    tokens, counts = returnDistribution(tokens_train)
    print('\n Distribution of most common tokens')

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
    print(f"Vocal size = n_clusters -> {vocab_size} = {n_clusters}")
    start_time = time.time()
    if loss == "NCE":
        skipgram_model, skipgram_history, epochEarlyStopped = train_skipgram_with_nce(train_token_sequences, vocab_size, embedding_dim, window_size_skipgram, epochs, num_negative_samples)
    else:
    # skipgram_model = train_skipgram(train_token_sequences, vocab_size, embedding_dim, window_size_skipgram, epochs, loss=loss)
        skipgram_model, skipgram_history, epochEarlyStopped = train_skipgram_withinSameConversations(train_token_sequences, vocab_size, embedding_dim, window_size_skipgram, epochs, loss=loss)
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

    SaveEmbeddingsToOutput02(filepath_data, trainValTest_embeddings, labels_all, subfoldername, formatted_datetime, indices_all, **name_kwargs)

    name_kwargs_train = { "train": "Set", **name_kwargs}
    name_kwargs_val = {"val": "Set", **name_kwargs}
    name_kwargs_test = {"test": "Set", **name_kwargs}
    SaveEmbeddingsToOutput02(filepath_data, train_embeddings, labels_train, subfoldername, formatted_datetime, indices_train, **name_kwargs_train)
    SaveEmbeddingsToOutput02(filepath_data, val_embeddings, labels_val, subfoldername, formatted_datetime, indices_val, **name_kwargs_val)
    SaveEmbeddingsToOutput02(filepath_data, test_embeddings, labels_test, subfoldername, formatted_datetime, indices_test, **name_kwargs_test)

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

    saveResultsFile02(config, filepath_data, filepath_labels, all_metrics, n_clusters_list, allDataShapes,
                    allSegmenthapes, skipgram_history, total_skipgram_time, epochEarlyStopped, tokens_train, subfoldername, formatted_datetime, **name_kwargs)

 #   plot_umap_of_segments(train_embeddings, labels_train_seq)
    setType = "NO"
    filenamePrefix_SilVsNCL = "SilVsNCL"
    filenamePrefix_metricsVsNCL = "MetricsVsNCL"
    filenamePrefix_TokensDistr = "TokensDistribution"
    filenamePrefix_TSNE = "TSNE"
    filenamePrefix_PCA = "PCA"
    filenamePrefix_UMAP = "UMAP"
    top_n = 20

    plotSilhouetteVsNClusters(n_clusters_list, all_metrics['silhouette'], filenamePrefix_SilVsNCL, filepath_data, subfoldername, formatted_datetime, setType, **name_kwargs)
    plot_clustering_metrics(all_metrics, filenamePrefix_metricsVsNCL, filepath_data, subfoldername, formatted_datetime, setType, **name_kwargs)
    plot_token_distribution_Bar(tokens_train, top_n, filenamePrefix_TokensDistr, filepath_data, subfoldername, formatted_datetime, setType, percent=True, **name_kwargs)
    #  plot_tsnePCAUMAP(TSNE, np.array(segments_train), labels_segments_train, config["perplexity"], config["random_state"], "of data_train", "no")
    plot_tsnePCAUMAP(TSNE, segments_train, kmeans_model.fit_predict(segments_train), config["perplexity"], f"of Segments Colored by KMeans Clusters (n={n_clusters})",
                     filenamePrefix_TSNE, filepath_data, subfoldername, formatted_datetime, setType,
                     kmeans_model, config["random_state"], "no", **name_kwargs)
    plot_tsnePCAUMAP(PCA, segments_train, kmeans_model.fit_predict(segments_train), config["perplexity"], f"of Segments Colored by KMeans Clusters (n={n_clusters})",
                     filenamePrefix_PCA, filepath_data, subfoldername, formatted_datetime, setType,
                     kmeans_model, config["random_state"], "no", **name_kwargs)
    plot_tsnePCAUMAP(UMAP, segments_train, kmeans_model.fit_predict(segments_train), config["perplexity"], f"of Segments Colored by KMeans Clusters (n={n_clusters})",
                     filenamePrefix_UMAP, filepath_data, subfoldername, formatted_datetime, setType,
                     kmeans_model, config["random_state"], "no", **name_kwargs)

    filenamePrefix_SGLossvsEpoch = "SGLossvsEpoch"
    if epochEarlyStopped is not None:
        epochBestWeights = epochEarlyStopped - config['early_stopping_patience']
    else:
        epochBestWeights = None
    plotSkipgramLossVsEpoch(skipgram_history, filenamePrefix_SGLossvsEpoch, filepath_data, subfoldername, formatted_datetime,
                            setType="NO", epochEarlyStopped=epochEarlyStopped, epochBestWeights=epochBestWeights, **name_kwargs)

    PlothelpDict = {
        "filepath_data": filepath_data,
        "subfoldername": subfoldername,
        "formatted_datetime": formatted_datetime,
        "setType": setType,
        "name_kwargs": name_kwargs
    }

    dataType = "initialData"
    analyze_all_embedding_plots(data_train, data_val, data_test, dataType, save=True, **PlothelpDict)
    dataType = "Embeddings"
    analyze_all_embedding_plots(train_embeddings, val_embeddings, test_embeddings, dataType, save=True, **PlothelpDict)

    agreement_train = compare_token_assignments("Train", segments_train, kmeans_model, knn_model, save=True, **PlothelpDict)
    agreement_val = compare_token_assignments("Validation", segments_val, kmeans_model, knn_model, save=True, **PlothelpDict)
    agreement_test = compare_token_assignments("Test", segments_test, kmeans_model, knn_model, save=True, **PlothelpDict)



filepath_data = "Pitt_data_mfcc_sR44100_hopL256_mfcc_summary_nMFCC13_nFFT512_2025-05-03_15-09-33.csv"
filepath_data = "Pitt_data_mfcc_sR44100_hopL256_mfcc_summaryTrue_use_mfcc_deltasTrue_nMFCC13_nFFT512_2025-05-03_20-05-09.csv"
filepath_data = "Pitt_data_mfcc_sR44100_hopL512_mfcc_summaryFalse_use_mfcc_deltasTrue_nMFCC13_nFFT2048_2025-05-04_00-05-37.csv"
filepath_data = "Pitt_data_mfcc_sR44100_hopL1024_mfcc_summaryFalse_use_mfcc_deltasFalse_nMFCC13_nFFT2048_2025-05-27_00-13-51.npy"
filepath_data = "Pitt_data_mfcc_sR88200_hopL1024_mfcc_summaryTrue_use_mfcc_deltasTrue_nMFCC13_nFFT2048_2025-05-27_08-31-38.npy"
filepath_data = "Pitt_data_audio_features_sR44100_hopL512_summarAudFFalse_nFFT1024_resampleTrue_2025-06-04_16-37-39.npy"
filepath_data = "Pitt_data_audio_features_sR44100_hopL512_summarAudFTrue_nFFT1024_2025-06-04_18-10-41.npy"
filepath_data = "Pitt_data_audio_features_sR44100_hopL512_summarAudFTrue_nFFT1024_2025-06-04_18-54-48.npy"
filepath_data = "Pitt_data_raw_sR44100_frameL1024_resampleTrue_2025-06-05_21-44-17.npy"
filepath_data = "Pitt_data_raw_sR88200_resampleTrue10000_2025-06-05_22-07-47.npy"
#filepath_data = "Pitt_data_mfcc_sR44100_hopL1024_mfcc_summaryFalse_use_mfcc_deltasFalse_nMFCC13_nFFT2048_2025-05-20_01-24-12.npy"

filepath_labels = "Pitt_labels_mfcc_sR44100_hopL256_mfcc_summary_nMFCC13_nFFT512_2025-05-03_15-09-33.csv"
filepath_labels = "Pitt_labels_mfcc_sR44100_hopL256_mfcc_summaryTrue_use_mfcc_deltasTrue_nMFCC13_nFFT512_2025-05-03_20-05-09.csv"
filepath_labels= "Pitt_labels_mfcc_sR44100_hopL512_mfcc_summaryFalse_use_mfcc_deltasTrue_nMFCC13_nFFT2048_2025-05-04_00-05-37.csv"
filepath_labels = "Pitt_labels_mfcc_sR44100_hopL1024_mfcc_summaryFalse_use_mfcc_deltasFalse_nMFCC13_nFFT2048_2025-05-27_00-13-51.csv"
filepath_labels = "Pitt_labels_mfcc_sR88200_hopL1024_mfcc_summaryTrue_use_mfcc_deltasTrue_nMFCC13_nFFT2048_2025-05-27_08-31-38.csv"
filepath_labels = "Pitt_labels_audio_features_sR44100_hopL512_summarAudFFalse_nFFT1024_resampleTrue_2025-06-04_16-37-39.csv"
filepath_labels = "Pitt_labels_audio_features_sR44100_hopL512_summarAudFTrue_nFFT1024_2025-06-04_18-10-41.csv"
filepath_labels = "Pitt_labels_audio_features_sR44100_hopL512_summarAudFTrue_nFFT1024_2025-06-04_18-54-48.csv"
filepath_labels = "Pitt_labels_raw_sR44100_frameL1024_resampleTrue_2025-06-05_21-44-17.csv"
filepath_labels = "Pitt_labels_raw_sR88200_resampleTrue10000_2025-06-05_22-07-47.csv"
#filepath_labels = "Pitt_labels_mfcc_sR44100_hopL1024_mfcc_summaryFalse_use_mfcc_deltasFalse_nMFCC13_nFFT2048_2025-05-20_01-24-12.csv"

# Configuration dictionary to store hyperparameters and settings
config = {
    "clusteringAlgorithm": "KMEANS",
    "dbscan_eps_list": [0.1, 0.5, 0.7, 1.0],
 #   "n_clusters_list": [2, 6],#, 8, 12, 16, 24, 32, 48],# 64, 78, 96, 128, 196, 256, 512],
    "n_clusters_list": [4, 6, 8, 12, 16],
 #   "n_clusters_list": [2, 4, 8, 16, 24, 32, 64, 128, 256, 300, 350], #500, 700, 1000],
    "knn_neighbors": 3,        # Number of neighbors for k-NN - 50
    "window_size": 512,    # 2       # Window size for sequence generation - 10
    "stride": 256,          # 8      # Stride for sequence generation - 1
    "embedding_dim": 3,       # Dimension of word embeddings - 300
    "window_size_skipgram": 40, # - 20
    "epochs": 50,                # Number of training epochs
    "num_negative_samples": 10,
    "optimizer_skipgram": 'adagrad', # Adagrad in nalmpantis paper
    "skipgram_loss": "NCE", # "sparse_categorical_crossentropy", "NCE
    "sequenceEmbeddingAveragingMethod": "Average", # "Average", "Weighted"
    "batch_size": 128,          # Batch size for training
    "perplexity": 30,           # t-SNE perplexity
    "random_state": 42,         # Random state for reproducibility
    "n_init": 50, #default 10 in kmeans. use 50 to help avoid collapse
    "output_folder": "02_Embeddings",  # Folder for saving embeddings
    "enable_scaling": True,
    "scaler": MinMaxScaler,  # MinMaxScaler(), StandardScaler(), "noScaling"
    "rowWiseScaling": True,
    "early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-4
}
# e.g. paper -> embedding dim = 300, window size = 6, etc

mainLogic()
import numpy as np
import seaborn as sns
import umap
import umap.umap_ as umap
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.signal import spectrogram
from collections import Counter

from utils00 import returnFileNameToSave

# ----- 01 -----
def plot_token_distribution_Histogram(data, name="Token", bins=30, title=None, save_path=None, show=True, stats=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram (blue bars)
    sns.histplot(
        data,
        bins=bins,
        color="#4A90E2",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.9,
        ax=ax,
        stat="density"  # Or "count" if using older seaborn
    )

    # Now overlay KDE separately (black line)
    sns.kdeplot(
        data,
        color="black",
        linewidth=2,
        ax=ax
    )

    ax.set_xlabel(name, fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.set_title(title or f"{name} Distribution", fontsize=16, weight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=16)

    if stats:
        legend_text = (
            f"Count: {stats['count']}\n"
            f"Mean: {stats['mean']:.2f} - Std: {stats['std']:.2f}\n"
            f"Min: {stats['min']:.2f} - Max: {stats['max']:.2f}"
        )
        ax.legend([legend_text], loc="upper right", fontsize=18, frameon=True, framealpha=0.9)

# ----- 01 -----
def plot_colName_distributions(df_metadata, colName="duration", labels=("ALL", "C", "D"),
                               title="Duration Distributions by Label", bins=50):
    """
    Plots duration histograms with KDE overlays per label from a metadata DataFrame.

    Parameters:
    - df_metadata: DataFrame with at least ["duration", "label"] columns
    - labels: Tuple or list of labels to plot (default: ["ALL", "C", "D"])
    - title: Plot title
    - bins: Number of histogram bins
    """
    fig, axs = plt.subplots(len(labels), 1, figsize=(8, 5 * len(labels)), gridspec_kw={'hspace': 0.1},
                            constrained_layout=True)

    labelStats_Dict = {}
    for i, label in enumerate(labels):
        if label == "ALL":
            subset = df_metadata
        else:
            subset = df_metadata[df_metadata["label"] == label]

        if not subset.empty:
            print(f"\nLabel: {label}")
            colName_Values = subset[colName]

            colName_mean = colName_Values.mean()
            colName_std = colName_Values.std()
            colName_count = colName_Values.count()
            colName_min = colName_Values.min()
            colName_max = colName_Values.max()

            stats = {
                "count": colName_count,
                "mean": colName_mean,
                "std": colName_std,
                "min": colName_min,
                "max": colName_max
            }
            labelStats_Dict[label] = stats

            print(
                f"{colName} Count: {colName_count}, Mean: {colName_mean:.2f}s, Std: {colName_std:.2f}s, Min: {colName_min:.2f}s, Max: {colName_max:.2f}s")

            plot_token_distribution_Histogram(
                data=colName_Values,
                name=f"{colName} (s)",
                bins=bins,
                title=f"{label} Labels",
                stats=stats,
                ax=axs[i]
            )

    percentile_99 = np.percentile(df_metadata[colName], 99)  # Calculate the percentile
    common_xlim = (0, df_metadata[colName].max())  # Normalize X-axis across plots
    for ax in axs:
        #    ax.set_xlim([0, percentile_99]) # Set x-axis limit to 99th percentile
        ax.set_xlim(common_xlim)

    plt.suptitle(title, fontsize=20, weight='bold')
    plt.show()

    return labelStats_Dict

# ----- 02 -----
def plotSilhouetteVsNClusters(n_clusters_list, all_Silhouettes):
    plt.figure(figsize=(8, 5))
    plt.plot(n_clusters_list, all_Silhouettes, marker='o', linestyle='-')
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----- 02 -----
def plot_token_waveforms(windows, labels, token_id, sample_rate=11025, n_samples=5):
    token_windows = windows[labels == token_id]
    if len(token_windows) == 0:
        print(f"No samples found for Token {token_id}")
        return

    plt.figure(figsize=(15, 4))
    for i in range(min(n_samples, len(token_windows))):
        t = np.linspace(0, len(token_windows[i]) / sample_rate, num=len(token_windows[i]))
        plt.subplot(1, n_samples, i + 1)
        plt.plot(t, token_windows[i])
        plt.title(f'Token {token_id} - Sample {i + 1}')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
    plt.suptitle(f'Waveforms for Token {token_id}')
    plt.show()

# ----- 02 -----
def plot_token_spectrograms(windows, labels, token_id, sample_rate=11025, n_samples=5):
    token_windows = windows[labels == token_id]
    if len(token_windows) == 0:
        print(f"No samples found for Token {token_id}")
        return

    plt.figure(figsize=(15, 4))
    for i in range(min(n_samples, len(token_windows))):
        f, t, Sxx = spectrogram(token_windows[i], fs=sample_rate)
        plt.subplot(1, n_samples, i + 1)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.title(f'Token {token_id} - Sample {i + 1}')
        plt.ylabel('Freq (Hz)')
        plt.xlabel('Time (s)')
        plt.tight_layout()
    plt.suptitle(f'Spectrograms for Token {token_id}')
    plt.show()

# ----- 02 -----
def plot_tsne_of_segments(segments, labels, perplexity=30, random_state=42):
    print("🌀 Running t-SNE for 2D projection...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    reduced = tsne.fit_transform(segments)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=3)
    plt.title("t-SNE Projection of Segments Colored by Token")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter, label="Token ID")
    plt.tight_layout()
    plt.show()

# ----- 02 -----
def plot_umap_of_segments(segments, labels):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(segments)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=3)
    plt.title("UMAP Projection of Segments Colored by Token")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(scatter, label="Token ID")
    plt.tight_layout()
    plt.show()

# ----- 02 -----
def plot_token_distribution_Bar(tokens, top_n=20):
    token_counts = Counter(tokens)
    top = token_counts.most_common(top_n)
    labels, values = zip(*top)
    plt.bar(labels, values)
    plt.title("Top Token Frequencies")
    plt.xlabel("Token ID")
    plt.ylabel("Count")
    plt.show()

# ----- 02 and 03 -----
def plot_tsnePCAUMAP(algorithm, data, labels, perplexity, random_state, title, remove_outliers=True):

    print(f"\n ----- Starting algorithm - {algorithm} -----")
  #  print("Variance of features:", np.var(data, axis=0))
    print("Class distribution:", np.bincount(labels))

    # Remove outliers
    if remove_outliers==True:
        original_len = len(data)

        z_scores = np.abs(zscore(data))
        mask = (z_scores < 3).all(axis=1)  # keep only data points within 3 std devs
        data = data[mask]
        labels = labels[mask]
        print("Filtered data shape:", data.shape)
        print("Filtered class distribution:", np.bincount(labels))
        removed = original_len - len(data)
        print(f"Removed {removed} outlier(s)")

    """Applies algorithm and plots results."""
    if algorithm == TSNE:
    #    pca = PCA(n_components=30, random_state=42)  # Reduce to 30 dimensions
    #    X_pca = pca.fit_transform(data)

        transformer_alg = TSNE(n_components=2, perplexity=perplexity, method='barnes_hut', max_iter=250, random_state=42)
  #      transformer_alg = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    elif algorithm == PCA:
        transformer_alg = PCA(n_components=2, random_state=random_state)
    elif algorithm == umap.UMAP:
        transformer_alg = umap.UMAP(n_components=2, random_state=random_state)
    else:
        raise ValueError("Invalid algorithm! Use TSNE, PCA, or UMAP.")
    transformed = transformer_alg.fit_transform(data)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=transformed[:, 0], y=transformed[:, 1], hue=labels, palette="viridis", alpha=0.6)
    plt.title(f"{algorithm.__name__} Visualization " + title)
    plt.xlabel(f"{algorithm.__name__} Component 1"); plt.ylabel(f"{algorithm.__name__} Component 2")
  #  plt.show()
    print(f" ----- Finished algorithm - {algorithm} -----")

# ----- 03 -----
def plotTrainValMetrics(history, filepath_data, figureNameParams, flagRegression = "NO"):
    # Access metrics from the history
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Get the number of epochs from the length of the accuracy history
    epochs = len(training_accuracy)

    # Create a 2x2 grid for subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot training and validation accuracy
    axes[0, 0].plot(range(1, epochs + 1), training_accuracy, label='Training Accuracy', color='blue', marker='o')
    axes[0, 0].plot(range(1, epochs + 1), validation_accuracy, label='Validation Accuracy', color='orange', marker='o')
    axes[0, 0].set_title('Accuracy vs Epoch')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot training and validation loss
    axes[0, 1].plot(range(1, epochs + 1), training_loss, label='Training Loss', color='blue', marker='o')
    axes[0, 1].plot(range(1, epochs + 1), validation_loss, label='Validation Loss', color='orange', marker='o')
    axes[0, 1].set_title('Loss vs Epoch')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Get all available metric names
    metric_keys = list(history.history.keys())

    # Find keys dynamically (handles variations like precision_1, precision_2, etc.)
    precision_key = next((k for k in metric_keys if 'precision' in k.lower()), None)
    recall_key = next((k for k in metric_keys if 'recall' in k.lower()), None)


    if flagRegression == "NO":
        # Extract metrics dynamically
        if precision_key and recall_key:
            training_precision = history.history[precision_key]
            validation_precision = history.history[f'val_{precision_key}']
            training_recall = history.history[recall_key]
            validation_recall = history.history[f'val_{recall_key}']
        else:
            training_precision = history.history['precision']
            validation_precision = history.history['val_precision']
            training_recall = history.history['recall']
            validation_recall = history.history['val_recall']

        axes[1, 0].plot(range(1, epochs + 1), training_precision, label='Training Precision', color='blue', marker='o')
        axes[1, 0].plot(range(1, epochs + 1), validation_precision, label='Validation Precision', color='orange', marker='o')
        axes[1, 0].set_title('Precision vs Epoch')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot training and validation MSE
        axes[1, 1].plot(range(1, epochs + 1), training_recall, label='Training Recall', color='blue', marker='o')
        axes[1, 1].plot(range(1, epochs + 1), validation_recall, label='Validation Recall', color='orange', marker='o')
        axes[1, 1].set_title('Recall vs Epoch')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        training_mae = history.history['mae']
        validation_mae = history.history['val_mae']
        training_mse = history.history['mse']
        validation_mse = history.history['val_mse']
        # Plot training and validation MAE
        axes[1, 0].plot(range(1, epochs + 1), training_mae, label='Training MAE', color='blue', marker='o')
        axes[1, 0].plot(range(1, epochs + 1), validation_mae, label='Validation MAE', color='orange', marker='o')
        axes[1, 0].set_title('MAE vs Epoch')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot training and validation MSE
        axes[1, 1].plot(range(1, epochs + 1), training_mse, label='Training MSE', color='blue', marker='o')
        axes[1, 1].plot(range(1, epochs + 1), validation_mse, label='Validation MSE', color='orange', marker='o')
        axes[1, 1].set_title('MSE vs Epoch')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    filenameFull = returnFileNameToSave(filepath_data, figureNameParams)

    plt.savefig(filenameFull)  # Save the plot using the dynamic filename
    print(filenameFull)

 #   plt.show()

# ----- 03 -----
def plot_bootstrap_distribution(bootstrap_accuracies, lower_bound, upper_bound):
    plt.hist(bootstrap_accuracies, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=2, label=f'Lower bound: {lower_bound:.2f}')
    plt.axvline(upper_bound, color='green', linestyle='dashed', linewidth=2, label=f'Upper bound: {upper_bound:.2f}')
    plt.axvline(np.mean(bootstrap_accuracies), color='orange', linestyle='dashed', linewidth=2,
                label=f'Mean: {np.mean(bootstrap_accuracies):.2f}')
    plt.legend()
    plt.title('Bootstrap Distribution of Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.show()
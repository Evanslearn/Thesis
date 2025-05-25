import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import umap.umap_ as umap
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.signal import spectrogram
from collections import Counter

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from utils00 import returnFileNameToSave


# ----- 01 -----
def plot_audio_feature(data, sr, feature_type='raw', hop_length=512):
    if feature_type == 'raw':
        # Plot raw waveform
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(data, sr=sr)
        plt.title('Raw Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()

    elif feature_type == 'mfcc':
        # Plot precomputed MFCCs
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(data, x_axis='time', sr=sr, hop_length=hop_length)
        plt.colorbar()
        plt.title('MFCC')
        plt.ylabel('MFCC Coefficient')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError(f"Unknown feature_type '{feature_type}'. Use 'raw' or 'mfcc'.")



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
        ax.legend(
            [legend_text],
            loc="upper center",  # start from upper center
            bbox_to_anchor=(0.75, 1),  # move slightly right
            fontsize=18,
            frameon=True,
            framealpha=0.9
        )

# ----- 01 -----
def plot_colName_distributions(df_metadata, colName="duration", labels=("ALL", "C", "D"),
                               title="Distribution by Label", bins=50):
    """
    Plots histograms of a given column, grouped by label, with auto-detected units.

    Parameters:
    - df_metadata: DataFrame with at least [colName, "label"] columns
    - labels: Tuple or list of labels to plot (default: ["ALL", "C", "D"])
    - title: Plot title
    - bins: Number of histogram bins
    """

    # Auto unit detection based on column name
    unit = "s" if "duration" in colName.lower() else "Hz" if any(k in colName.lower() for k in ["freq", "rate"]) else ""

    fig, axs = plt.subplots(len(labels), 1, figsize=(8, 5 * len(labels)), gridspec_kw={'hspace': 0.1},
                            constrained_layout=True)

    labelStats_Dict = {}

    for i, label in enumerate(labels):
        subset = df_metadata if label == "ALL" else df_metadata[df_metadata["label"] == label]

        if not subset.empty:
            print(f"\nLabel: {label}")
            col_values = subset[colName]

            stats = {
                "count": col_values.count(),
                "mean": col_values.mean(),
                "std": col_values.std(),
                "min": col_values.min(),
                "max": col_values.max()
            }
            labelStats_Dict[label] = stats

            # Nicely formatted stats output
            print(
                f"{colName} Count: {stats['count']}, "
                f"Mean: {stats['mean']:.2f}{unit}, "
                f"Std: {stats['std']:.2f}{unit}, "
                f"Min: {stats['min']:.2f}{unit}, "
                f"Max: {stats['max']:.2f}{unit}"
            )

            plot_token_distribution_Histogram(
                data=col_values,
                name=f"{colName} ({unit})" if unit else colName,
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

# ----- HELP -----

def saveFigure02(fig, filenamePrefix, filepath_data, subfoldername, formatted_datetime, setType="NO", **kwargs):
    case_type = "Pitt" if "Pitt" in filepath_data else "Lu"
    case_type_str = f"{case_type}_{setType}" if setType != "NO" else f"_{case_type}_"

    filename_variables = "".join(
        f"{key}{value}".replace("{", "").replace("}", "") + "_" for key, value in kwargs.items()).rstrip("_")

    # Helper function to generate paths dynamically
    def generate_path(prefix):
        return f"{subfoldername}/{prefix}_{case_type_str}{filename_variables}_{formatted_datetime}.png"

    figureFullName = generate_path(filenamePrefix)

    fig.savefig(figureFullName)
    print(f"Saved {filenamePrefix} plot to:", figureFullName)

# ----- 02 -----
def plotSilhouetteVsNClusters(n_clusters_list, all_Silhouettes, filenamePrefix, filepath_data, subfoldername, formatted_datetime, setType="NO", **kwargs):
    # Create a 2x2 grid for subplots
    fig = plt.figure(figsize=(8, 5))
    plt.plot(n_clusters_list, all_Silhouettes, marker='o', linestyle='-')
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
  #  plt.show()
    saveFigure02(fig, filenamePrefix, filepath_data, subfoldername, formatted_datetime, setType, **kwargs)

# ----- 02 -----
def plot_clustering_metrics(metrics_dict, filenamePrefix, filepath_data, subfoldername, formatted_datetime, setType, **kwargs):
    clusters = metrics_dict["n_clusters"]
    fig = plt.figure(figsize=(6, 12))  # Taller figure for vertical layout

    plt.subplot(3, 1, 1)
    plt.plot(clusters, metrics_dict["silhouette"], marker='o')
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")

    plt.subplot(3, 1, 2)
    plt.plot(clusters, metrics_dict["ch_index"], marker='o')
    plt.title("Calinski-Harabasz Index vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Calinski-Harabasz Index")

    plt.subplot(3, 1, 3)
    plt.plot(clusters, metrics_dict["db_index"], marker='o')
    plt.title("Davies-Bouldin Index vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Davies-Bouldin Index")

    plt.tight_layout()
  #  plt.show()

    saveFigure02(fig, filenamePrefix, filepath_data, subfoldername, formatted_datetime, setType, **kwargs)


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
def plot_token_distribution_Bar(tokens, top_n, filenamePrefix, filepath_data, subfoldername, formatted_datetime, setType, percent=False,  **kwargs):
    token_counts = Counter(tokens)
    total_count = sum(token_counts.values())
    top = token_counts.most_common(top_n)

    labels, values = zip(*top)

    if percent:
        values = [v / total_count * 100 for v in values]
        ylabel = r"Percent (%)"  # <- this fixes the issue
        value_fmt = lambda v: f"{v:.1f}%"
        legend_label = f"Total tokens: {total_count} (100%)"
    else:
        ylabel = "Count"
        value_fmt = lambda v: str(int(v))
        legend_label = f"Total tokens: {total_count}"

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values)

    ax.set_title("Top Token Frequencies")
    ax.set_xlabel("Token ID")
    ax.set_ylabel(ylabel)

    # Add value labels above bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + max(values) * 0.01,  # small vertical offset
            f"{yval:.1f}%" if percent else f"{int(yval)}",
            ha='center', va='bottom', fontsize=9
        )

    # Add legend with total count/%
    ax.legend([legend_label], loc='upper right')
    plt.tight_layout()
    plt.show()

    saveFigure02(fig, filenamePrefix, filepath_data, subfoldername, formatted_datetime, setType, **kwargs)


def plotEarlyStopLines(epochEarlyStopped, epochBestWeights, loss_values, ax=None):
    plot_func = ax if ax is not None else plt

    if epochEarlyStopped is not None and epochEarlyStopped <= len(loss_values):
        plot_func.axvline(x=epochEarlyStopped, color='red', linestyle='--', label=f"Early Stop  @ Epoch {epochEarlyStopped}")
        plot_func.scatter(epochEarlyStopped, loss_values[epochEarlyStopped - 1], color='red', zorder=5)

    if epochBestWeights is not None and epochBestWeights <= len(loss_values):
        plot_func.axvline(x=epochBestWeights, color='green', linestyle='--', label=f"Checkpoint @ Epoch {epochBestWeights}")
        plot_func.scatter(epochBestWeights, loss_values[epochBestWeights - 1], color='green', zorder=5)

# ----- 02 -----
def plotSkipgramLossVsEpoch(skipgram_history, filenamePrefix, filepath_data, subfoldername, formatted_datetime,
                            setType="NO", epochEarlyStopped=None, epochBestWeights=None, **kwargs):
    if isinstance(skipgram_history, dict):
        df_history = pd.DataFrame(skipgram_history)
    else:
        df_history = pd.DataFrame(skipgram_history.history)
    df_history.insert(0, "Epoch", range(1, len(df_history) + 1))  # Add epoch numbers
    df_history = df_history.round(6)
    loss_values = df_history['loss']

    fig = plt.figure(figsize=(8, 5))
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, marker='o', linestyle='-', label="Training Loss")
    plt.title("Skip-Gram Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plotEarlyStopLines(epochEarlyStopped, epochBestWeights, loss_values)

    plt.legend()
    plt.tight_layout()

    saveFigure02(fig, filenamePrefix, filepath_data, subfoldername, formatted_datetime, setType, **kwargs)
    plt.show()






import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist

def analyze_all_embedding_plots(train_embeddings, val_embeddings, test_embeddings, dataType, save=False, **PlothelpDict):
    def save_fig(fig, filenamePrefix):
        if save:
            saveFigure02(fig, filenamePrefix, PlothelpDict['filepath_data'], PlothelpDict['subfoldername'], PlothelpDict['formatted_datetime'], PlothelpDict['setType'], **PlothelpDict['name_kwargs'])

    sets = [("Train", train_embeddings), ("Validation", val_embeddings), ("Test", test_embeddings)]

    # ---- PCA Explained Variance ----
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, (set_name, emb) in zip(axes, sets):
        pca = PCA()
        pca.fit(emb)
        explained_var = np.cumsum(pca.explained_variance_ratio_)

        n_dims = len(explained_var)
        top_k = min(10, n_dims)
        top10_var = np.sum(pca.explained_variance_ratio_[:top_k])
        ax.set_title(f"{set_name} - PCA Variance ({top_k} dims ≈ {top10_var:.2%})")

        ax.plot(np.arange(1, n_dims + 1), explained_var, marker='o')
        ax.set_xscale('linear')
        plt.xlabel("PCA Component")

        ax.set_ylabel("Cumulative Explained Variance")
        ax.grid(True)
        print(f"📊 {set_name} - % Variance in first {top_k} PCA components: {top10_var:.4f}")

        threshold = 0.90
        dim_90 = np.where(explained_var >= threshold)[0][0] + 1  # +1 for 1-based index
        var_at_90 = explained_var[dim_90 - 1]
        ax.axvline(dim_90, linestyle='--', color='red', label=f"First ≥90%: PC {dim_90} ({var_at_90:.1%})")
        ax.legend()

    axes[-1].set_xlabel("Number of Components")
    fig.suptitle("PCA Variance Across Sets", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, f"pca_variance{dataType}")
    plt.show()

    # ---- Cosine Similarity Heatmaps (3x1 or 1x3 layout, shared colorbar, fewer ticks) ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(wspace=0.3, right=0.88)

    cbar_ax = fig.add_axes([0.89, 0.25, 0.015, 0.5])  # better aligned with heatmaps
    for idx, (ax, (set_name, emb)) in enumerate(zip(axes, sets)):
        sim_matrix = cosine_similarity(emb)

        # Limit number of ticks
        n = sim_matrix.shape[0]
        tick_step = max(n // 10, 1)
        tick_positions = np.arange(0, n, tick_step)

        sns.heatmap(sim_matrix, ax=ax, cmap='viridis', cbar=(idx == 0),  cbar_ax=(cbar_ax if idx == 0 else None),
            square=True)

        ax.set_title(f"{set_name} - Cosine Similarity")
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_positions, rotation=0)
        ax.set_yticklabels(tick_positions, rotation=0)

    fig.suptitle("Cosine Similarity Heatmaps Across Sets", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    save_fig(fig, f"cosine_similarity{dataType}")
    plt.show()

    # ---- Histogram of Pairwise Cosine Distances ----
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, (set_name, emb) in zip(axes, sets):
        distances = pdist(emb, metric='cosine')
        mean_d = np.mean(distances)
        std_d = np.std(distances)
        ax.hist(distances, bins=40, color='skyblue', edgecolor='black')
        ax.set_title(f"{set_name} - Cosine Distance Histogram")
        label = f"{set_name}\nμ={mean_d:.4f}\nσ={std_d:.4f}"
        ax.hist(distances, bins=40, color='skyblue', edgecolor='black', label=label)
        ax.legend()
        ax.set_ylabel("Frequency")
        ax.grid(True)
        print(f"📏 {set_name} - Mean cosine distance: {mean_d:.4f}, Std: {std_d:.4f}")

    axes[-1].set_xlabel("Cosine Distance")
    fig.suptitle("Cosine Distance Histograms Across Sets", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig,f"cosine_distance_hist{dataType}")
    plt.show()

# ----- 02 and 03 -----
def plot_tsnePCAUMAP(algorithm, data, labels, perplexity, title,
                     filenamePrefix, filepath_data, subfoldername, formatted_datetime, setType,
                     kmeans_model=None, random_state=42, remove_outliers=True, **kwargs):

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
        transformer = TSNE(n_components=2, perplexity=perplexity, method='barnes_hut', max_iter=250, random_state=random_state)
        if kmeans_model is not None:
            combined_data = np.vstack([data, kmeans_model.cluster_centers_])
            transformed_all = transformer.fit_transform(combined_data)
            transformed, centroids = transformed_all[:-kmeans_model.n_clusters], transformed_all[-kmeans_model.n_clusters:]
        else:
            transformed = transformer.fit_transform(data)
    elif algorithm == PCA:
        transformer = PCA(n_components=2, random_state=random_state)
        transformed = transformer.fit_transform(data)
        if kmeans_model is not None:
            centroids = transformer.transform(kmeans_model.cluster_centers_)
    elif algorithm == umap.UMAP:
        transformer = umap.UMAP(n_components=2, random_state=random_state)
        transformed = transformer.fit_transform(data)
        if kmeans_model is not None:
            centroids = transformer.transform(kmeans_model.cluster_centers_)
    else:
        raise ValueError("Invalid algorithm. Use TSNE, PCA, or UMAP.")

    # Create legend labels with sample counts
    counts = Counter(labels)
    label_map = {k: f"{k} (n={v})" for k, v in counts.items()}
    label_names = [label_map[l] for l in labels]

    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(x=transformed[:, 0], y=transformed[:, 1], hue=label_names, palette="viridis", alpha=0.6,
                    legend="full")

    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=120, marker='X', label='Cluster\nCentroid')

    plt.title(f"{algorithm.__name__} Visualization " + title)
    plt.xlabel(f"{algorithm.__name__} Component 1")
    plt.ylabel(f"{algorithm.__name__} Component 2")
    plt.legend(title="Cluster", fontsize='small', title_fontsize='medium', loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
#    plt.show()
    print(f" ----- Finished algorithm - {algorithm} -----")
    saveFigure02(fig, filenamePrefix, filepath_data, subfoldername, formatted_datetime, setType, **kwargs)

# ----- 03 -----
def plot_cosine_similarity_histogram(cosine_scores, description=""):
    plt.figure(figsize=(8, 6))
    plt.hist(cosine_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'Cosine similarity between individual vectors ({description})')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# ----- 03 -----
def plotClassBarPlots(Y_train, Y_val, Y_test):
    label_sets = [Y_train, Y_val, Y_test]
    titles = ['Train', 'Validation', 'Test']

    # A4 width ~11.7 inches, height adjusted for clarity
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 4), sharey=True)
    for i, (ax, labels, title) in enumerate(zip(axes, label_sets, titles)):
        unique, counts = np.unique(labels, return_counts=True)
        total = counts.sum()
        ratios = counts / total

        bars = ax.bar(['Class 0', 'Class 1'], counts, color=['#1f77b4', '#ff7f0e'])
        ax.set_title(f'{title} Class Distribution', fontsize=12)
        ax.set_xlabel('Class', fontsize=10)
        if i == 0:
            ax.set_ylabel('Count', fontsize=10)

        # Annotate with count and ratio above bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,  # vertical center
                f'{counts[j]}\n({ratios[j] * 100:.1f}%)',
                ha='center',
                va='center',
                fontsize=12,
                color='white',# if height > 40 else 'black',  # readable contrast
                fontweight='bold'
            )
    plt.tight_layout()
  #  plt.show()

# ----- 03 -----
def plotTrainValMetrics(history, filepath_data, figureNameParams, epochEarlyStopped=None, epochBestWeights=None, flagRegression = "NO"):
    # Access metrics from the history
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    # Get the number of epochs from the length of the accuracy history
    epochs = len(training_accuracy)

    # === Create separate figure for Loss ===
    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.plot(range(1, epochs + 1), training_loss, label='Training Loss', color='blue', marker='o')
    ax_loss.plot(range(1, epochs + 1), validation_loss, label='Validation Loss', color='orange', marker='o')
    ax_loss.set_title('Loss vs Epoch')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    plotEarlyStopLines(epochEarlyStopped, epochBestWeights, training_loss, ax_loss)
    ax_loss.legend()
    ax_loss.grid(True)
    fig_loss.tight_layout()

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
        # F1 score
        training_f1 = history.history.get('f1_score_keras', [])
        validation_f1 = history.history.get('val_f1_score_keras', [])


        axes[0, 1].plot(range(1, epochs + 1), training_precision, label='Training Precision', color='blue', marker='o')
        axes[0, 1].plot(range(1, epochs + 1), validation_precision, label='Validation Precision', color='orange', marker='o')
        axes[0, 1].set_title('Precision vs Epoch')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(range(1, epochs + 1), training_recall, label='Training Recall', color='blue', marker='o')
        axes[1, 0].plot(range(1, epochs + 1), validation_recall, label='Validation Recall', color='orange', marker='o')
        axes[1, 0].set_title('Recall vs Epoch')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(range(1, epochs + 1), training_f1, label='Training F1 Score', color='blue', marker='o')
        axes[1, 1].plot(range(1, epochs + 1), validation_f1, label='Validation F1 Score', color='orange', marker='o')
        axes[1, 1].set_title('F1 Score vs Epoch')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('F1 Score')
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

    for i, ax in enumerate(axes.flatten()):
  #      ax.legend(loc='upper right')
        plotEarlyStopLines(epochEarlyStopped, epochBestWeights, training_loss, ax)

    # Create custom legend handles
    custom_lines = []
    if epochEarlyStopped:
        custom_lines.append(Line2D([0], [0], color='red', linestyle='--', label=f"Early Stop  @ Epoch {epochEarlyStopped}"))
    if epochBestWeights:
        custom_lines.append(Line2D([0], [0], color='green', linestyle='--', label=f"Checkpoint @ Epoch {epochBestWeights}"))

    # Add global legend with only those handles
 #   fig.legend(handles=custom_lines, loc='center', ncol=1, fontsize='small', frameon=True)
    #   plt.subplots_adjust(bottom=0.15)
    if epochEarlyStopped != None and epochBestWeights != None:
        fig.legend(handles=custom_lines, loc='center', ncol=1, fontsize='small', frameon=True)
 #       plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave room on the right
 #   else:
 #       plt.tight_layout()
    plt.tight_layout()

    # Save both figures
    filenameFull_main = returnFileNameToSave(filepath_data, figureNameParams)
    filenameFull_loss = filenameFull_main.replace("figure_", "Lossfig")

    fig.savefig(filenameFull_main)
    fig_loss.savefig(filenameFull_loss)

    print("Saved metrics plot to:", filenameFull_main)
    print("Saved loss plot to:", filenameFull_loss)

  #  plt.show()

# ----- 03 -----
def calculateAndReturnConfusionMatrix(Y, Y_preds):
    cm_raw = confusion_matrix(Y, Y_preds)
    cm_norm = confusion_matrix(Y, Y_preds, normalize='true')
    return cm_raw, cm_norm

# ----- 03 -----
def plotAndSaveConfusionMatrix(cm_raw, cm_norm, filepath_data, figureNameParams):
    labels = ["Control", "Dementia"]

    disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=labels)
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)

    # Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    disp_raw.plot(ax=ax1, cmap='Blues', values_format='d', text_kw={'fontsize': 18})
    ax1.set_title("Confusion Matrix (Raw Counts)")

    disp_norm.plot(ax=ax2, cmap='Blues', values_format='.2f', text_kw={'fontsize': 18})
    ax2.set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()
   # plt.show()
    filename_cm = returnFileNameToSave(filepath_data, figureNameParams)
    filename_cm = filename_cm.replace("figure_", "ConfMatrix_")

    fig.savefig(filename_cm)
    print(f"Saved combined confusion matrix to: {filename_cm}")
    return

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


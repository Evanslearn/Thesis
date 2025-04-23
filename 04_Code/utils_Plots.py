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

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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
def plot_clustering_metrics(metrics_dict):
    clusters = metrics_dict["n_clusters"]
    plt.figure(figsize=(6, 12))  # Taller figure for vertical layout

    plt.subplot(3, 1, 1)
    plt.plot(clusters, metrics_dict["silhouette"], marker='o')
    plt.title("Silhouette Score")
    plt.xlabel("n_clusters")

    plt.subplot(3, 1, 2)
    plt.plot(clusters, metrics_dict["ch_index"], marker='o')
    plt.title("Calinski-Harabasz Index")
    plt.xlabel("n_clusters")

    plt.subplot(3, 1, 3)
    plt.plot(clusters, metrics_dict["db_index"], marker='o')
    plt.title("Davies-Bouldin Index (lower is better)")
    plt.xlabel("n_clusters")

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
def plot_tsnePCAUMAP(algorithm, data, labels, perplexity, title, random_state=42, remove_outliers=True):

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
    plt.show()
    print(f" ----- Finished algorithm - {algorithm} -----")

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
                color='white' if height > 40 else 'black',  # readable contrast
                fontweight='bold'
            )
    plt.tight_layout()
    plt.show()

# ----- 03 -----
def plotTrainValMetrics(history, filepath_data, figureNameParams, flagRegression = "NO"):
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
        training_f1 = history.history.get('f1_score', [])
        validation_f1 = history.history.get('val_f1_score', [])


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

    # Adjust layout to prevent overlap
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

    disp_raw.plot(ax=ax1, cmap='Blues', values_format='d')
    ax1.set_title("Confusion Matrix (Raw Counts)")

    disp_norm.plot(ax=ax2, cmap='Blues', values_format='.2f')
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


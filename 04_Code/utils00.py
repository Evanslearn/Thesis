import json
import os
from datetime import datetime
import random

import numpy as np
import pandas as pd
import umap
from matplotlib import pyplot as plt
from pandas._testing import iloc
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.manifold import TSNE
import umap.umap_ as umap

def returnFormattedDateTimeNow():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def makeLabelsInt(labels):
    # print(labels)
    return labels.map({'C': 0, 'D': 1}).to_numpy()

def returnFilepathToSubfolder(filename, subfolderName):

    # Get the current directory of script execution
    current_directory = os.getcwd()

    # Define the output folder inside the current directory
    output_folder = os.path.join(current_directory, subfolderName)

    # Create the folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the file path inside the output folder
    file_path = os.path.join(output_folder, filename)

    return file_path


def return_scaler_type(scaler, enable_scaling):
    if "MinMax" in scaler:
        scalerName = "MinMax"
    elif "Standard" in scaler:
        scalerName = "Standard"

    if enable_scaling != True:
        scalerName = "NoScaler"
    return scalerName

def readCsvAsDataframe(abspath, filepath, dataFilename = "data", as_series=False):
    full_path = abspath + filepath
    print(f"Reading file: {full_path}")

    if as_series:
        df = pd.read_csv(full_path, header=None).iloc[:, 0]
        print(f"{dataFilename} shape = {df.shape}")
        return df

    # For non-series, load line by line as arrays (handles variable-length time series)
    time_series_list = []
    with open(full_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split(",")))
            time_series_list.append(np.array(values))

    # Convert to DataFrame of dtype=object (non-rectangular)
    df = pd.DataFrame(time_series_list, dtype=object)
    print(f"{dataFilename} shape = {df.shape}")
    return df

def returnDataAndLabelsWithoutNA(data, labels, addIndexColumn = False):

    combined = pd.concat([data, labels.rename("label")], axis=1)
    combined = combined.dropna().reset_index(drop=True)

    data = combined.drop(columns="label")
    if addIndexColumn == True:
        data["index"] = data.index
    return data, combined["label"]

def doTrainValTestSplit(X_data, Y_targets, test_val_ratio = 0.3, valRatio_fromTestVal = 0.5, random_state = 0):
    # Create indices for the data
    indices = np.arange(len(X_data))

    # First split: Train vs (Test + Val)
    X_train, X_test_val, Y_train, Y_test_val, indices_train, indices_test_val = train_test_split(X_data, Y_targets, indices, test_size=test_val_ratio,
                                                                random_state=random_state, stratify=Y_targets)
    val_ratio = test_val_ratio * valRatio_fromTestVal
    print(
        f'''We have used {test_val_ratio * 100}% of the data for the test+val set. So now, the val_ratio = {valRatio_fromTestVal * 100}%
    of the val-test data, translates to {val_ratio * 100}% of the total data.''')

    # Second split: Test vs Val
    X_test, X_val, Y_test, Y_val, indices_test, indices_val = train_test_split(X_test_val, Y_test_val, indices_test_val, test_size=valRatio_fromTestVal,
                                                    random_state=random_state, stratify=Y_test_val)

    print(f"Train: {len(Y_train)}, \nVal: {len(Y_val)}, \nTest: {len(Y_test)}")

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, val_ratio, indices_train, indices_val, indices_test

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

def returnFileNameToSave(filepath_data, fileNameParams, imageflag = "YES"):
    # Extract the part after "Embeddings_" and remove the extension
    filename = os.path.basename(filepath_data)  # Get the base filename
    filename_without_extension = os.path.splitext(filename)[0]  # Remove the extension (.csv)
    dynamic_filename = filename_without_extension.replace("Embeddings_", "")  # Remove "Embeddings_"

    # Remove the old timestamp (assuming it's always at the end, separated by "_")
    parts = dynamic_filename.rsplit("_", 1)  # Split into two parts: before timestamp, and timestamp
    dynamic_filename_without_timestamp = parts[0]  # Keep only the first part

    # Generate the new timestamp
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_dynamic_filename = f"{dynamic_filename_without_timestamp}_{fileNameParams}_{current_timestamp}"

    if imageflag == "YES":
        fileExtension = "png"
        save_filename = f"figure_{new_dynamic_filename}.{fileExtension}"  # Save as PNG
    else:
        fileExtension = "csv"
        save_filename = f"metrics_{new_dynamic_filename}.{fileExtension}"  # Save as PNG
    # Define the new filename for saving the plot

    subfolderName = "03_ClassificationResults"
    filenameFull = returnFilepathToSubfolder(save_filename, subfolderName)
    return filenameFull

def saveTrainingMetricsToFile(history, model, config, learning_rate, optimizer, training_time, test_metrics, filepathsAll, predictions, actual_labels, filepath_data, fileNameParams, ratio_0_to_1_ALL):
    filenameFull = returnFileNameToSave(filepath_data, fileNameParams, imageflag="NO")

    # Convert history.history (dictionary) to DataFrame
    df_history = pd.DataFrame(history.history)
    df_history.insert(0, "Epoch", range(1, len(df_history) + 1)) # Add epoch numbers
    df_history = df_history.round(6)

    # Convert test results to DataFrame
    df_results = pd.DataFrame({
        "Prediction": predictions,
        "Actual Label": actual_labels
    })

    # Save everything into a single CSV file
    with open(filenameFull, "w") as f:
        f.write("Training History:\n")
        df_history.to_csv(f, index=False)

        f.write("\nRatio of 0s/1s Train-Val-Test:\n")
        for i in range(0,len(ratio_0_to_1_ALL)):
            f.write(f"{ratio_0_to_1_ALL[i]}\n")

        # Convert non-serializable objects to strings
        config_serializable = config.copy()
        config_serializable["metrics"] = [str(m) if not isinstance(m, str) else m for m in config["metrics"]]
        config_serializable["optimizer"] = str(config["optimizer"])  # Convert optimizer to string
        config_serializable["kernel_regularizer_dense"] = str(config["kernel_regularizer_dense"])  # Convert L2
        config_serializable["scaler"] = str(config.get("scaler", ""))
        f.write("THE WHOLE CONFIG FOLLLOWS:")
        json.dump(config_serializable, f, indent=4)

        f.write("\nTHE FILE PATHS FOLLOW:")
        json.dump(filepathsAll, f, indent=4)

        f.write(f"\nlearning_rate = {learning_rate}, Optimizer = {optimizer}")

        f.write("\nModel Architecture:\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))

        f.write("\nTraining Time:\n")
        f.write(f"{training_time:.6f} seconds\n")

        f.write("\nTest Set Metrics:\n")
        for metric_name, metric_value in test_metrics.items():
            f.write(f"{metric_name},{metric_value}\n")

        f.write("\nPredictions vs Actual Labels:\n")
        df_results.to_csv(f, index=False)

        f.write("\nRandom Sample Comparisons:\n")
        rand_index_pred = 5
        random_numbers = [random.randint(0, actual_labels.shape[0] - 1) for _ in range(rand_index_pred)]
        for i in random_numbers:
            f.write(f"For i = {i}, we have:\n")
            f.write(f"Y_predictions[i]     = {predictions[i]}\n")
            f.write(f"Y_test_normalized[i] = {actual_labels[i]}\n\n")

    print(f"All results saved to {filenameFull}!")
    return filenameFull

import re
def trim_datetime_suffix(common_part):
    return re.sub(r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$', '', common_part)

def returnDistribution(dataToCount, name="Token", file=None, display=True):
    output_lines = [f"{name} Distribution:"]
    tokens, counts = np.unique(dataToCount, return_counts=True)
    for token, count in zip(tokens, counts):
        output_lines.append(f"{name} {token}: {count} instances")

    # Output to file if file is given, else print to terminal
    if file:
        for line in output_lines:
            file.write(line + "\n")
    else:
        if display:
            for line in output_lines:
                print(line)

    return tokens, counts

def dropInstancesUntilClassesBalance(data, labels):
    labelClasses, counts = returnDistribution(labels, name="Label")
    count_0 = counts[0]
    count_1 = counts[1]

    # Step 2: Determine which label is the majority class
    majority_label = 0 if count_0 > count_1 else 1
    minority_count = min(count_0, count_1)
    majority_count = max(count_0, count_1)

    # Step 3: Find indices of the majority class
    labels_array = np.array(labels)
    majority_indices = np.where(labels_array == majority_label)[0]

    # Step 4: Randomly select excess indices to drop
    num_to_drop = majority_count - minority_count
    indices_to_drop = random.sample(list(majority_indices), num_to_drop)

    # Step 5: Create mask to keep the rest
    keep_mask = np.ones(len(labels), dtype=bool)
    keep_mask[indices_to_drop] = False

    # Apply mask to labels and any aligned array (e.g., data)
    labels_balanced = labels_array[keep_mask]
    data_balanced = data.iloc[keep_mask].reset_index(drop=True)
    print(f"Balanced class counts: {np.unique(labels_balanced, return_counts=True)}")

    return data_balanced, labels_balanced

def plot_token_distribution(data, name="Token", bins=30, title=None, save_path=None, show=True, stats=None, ax=None):
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
            f"Mean: {stats['mean']:.2f}\n"
            f"Std: {stats['std']:.2f}\n"
            f"Count: {stats['count']}"
        )
        ax.legend([legend_text], loc="upper right", fontsize=18, frameon=True, framealpha=0.9)


def read_padded_csv_with_lengths(filepath, pad_value=0.0):
    rows = []
    lengths = []
    max_len = 0

    with open(filepath, 'r') as f:
        for line in f:
            row = [float(val) for val in line.strip().split(',') if val]
            lengths.append(len(row))
            max_len = max(max_len, len(row))
            rows.append(row)

    padded_rows = [row + [pad_value] * (max_len - len(row)) for row in rows]
    df = pd.DataFrame(padded_rows)
    return df, lengths


import os
import soundfile as sf

def extract_duration_and_samplerate(labeled_files, verbose=True):
    """
    Returns a list of tuples: (filename, duration_sec, sample_rate, label)
    Safely extracts duration and sampling rate without loading full audio.
    """
    metadata = []

    for idx, (mp3_file, label) in enumerate(labeled_files):
        if verbose and idx % 20 == 0:
            print(f"🔍 Checking file #{idx}: {mp3_file}")

        try:
            info = sf.info(mp3_file)
            sr = info.samplerate
            duration_sec = info.frames / sr

            metadata.append((
                os.path.basename(mp3_file),
                round(duration_sec, 2),
                sr,
                label
            ))

        except Exception as e:
            print(f"❌ Failed to read {mp3_file}: {e}")
            metadata.append((
                os.path.basename(mp3_file),
                "ERROR",
                "ERROR",
                label
            ))

    return metadata



def plot_colName_distributions(df_metadata, colName="duration", labels=("ALL", "C", "D"), title="Duration Distributions by Label", bins=50):
    """
    Plots duration histograms with KDE overlays per label from a metadata DataFrame.

    Parameters:
    - df_metadata: DataFrame with at least ["duration", "label"] columns
    - labels: Tuple or list of labels to plot (default: ["ALL", "C", "D"])
    - title: Plot title
    - bins: Number of histogram bins
    """
    fig, axs = plt.subplots(len(labels), 1, figsize=(8, 5 * len(labels)), gridspec_kw={'hspace': 0.1}, constrained_layout=True)

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

            print(f"{colName} Mean: {colName_mean:.2f}s, Std: {colName_std:.2f}s, Count: {colName_count}")

            plot_token_distribution(
                data=colName_Values,
                name=f"{colName} (s)",
                bins=bins,
                title=f"{label} Labels",
                stats={
                    "mean": colName_mean,
                    "std": colName_std,
                    "count": colName_count
                },
                ax=axs[i]
            )


    percentile_99 = np.percentile(df_metadata[colName], 99) # Calculate the percentile
    common_xlim = (0, df_metadata[colName].max()) # Normalize X-axis across plots
    for ax in axs:
        ax.set_xlim([0, percentile_99]) # Set x-axis limit to 99th percentile
   #     ax.set_xlim(common_xlim)

    plt.suptitle(title, fontsize=20, weight='bold')
    plt.show()

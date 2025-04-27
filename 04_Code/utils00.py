import csv
import json
import os
from collections import Counter
from datetime import datetime
import re
import librosa
import time
import random
import numpy as np
import pandas as pd
from pandas._testing import iloc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def returnFormattedDateTimeNow():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def makeLabelsInt(labels):
    # print(labels)
    return labels.map({'C': 0, 'D': 1}).to_numpy()

def printLabelCounts(labels):
    counts = Counter(labels)
    print(f"Label Counts: {dict(counts)}")

def print_data_info(data, labels, stage=""):
    print(f"----- {stage} -----")
    print(f"Labels shape = {labels.shape}, Data shape = {data.shape}")

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

def doTrainValTestSplit(X_data, Y_targets, test_val_ratio = 0.3, valRatio_fromTestVal = 0.5, random_state = 42):
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


def trim_datetime_suffix(common_part):
    return re.sub(r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$', '', common_part)
def returnDistribution(dataToCount, name="Token", file=None, display=True):
    output_lines = [f"{name} Distribution:"]
    tokens, counts = np.unique(dataToCount, return_counts=True)
    for token, count in zip(tokens, counts):
        output_lines.append(f"{name} {token}: {count} instances")

    # Output to file if file is given, else print to terminal
    if file:
        file.write("\n")
        for line in output_lines:
            file.write(line + "\n")
    else:
        if display:
            print("\n")
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

def read_padded_csv_with_lengths(filepath, pad_value=0.0):
    # is used to load variable-length time series stored as CSV rows of unequal length, and outputs:
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

def returnL2Distance(data, labels):
    mean_0 = data[labels == 0].mean(axis=0)
    mean_1 = data[labels == 1].mean(axis=0)

    dist = np.linalg.norm(mean_0 - mean_1)
    print(f"L2 distance between class 0 and 1 mean vectors: {dist:.4f}")
    return dist

def cosine_similarity_between_means(data, labels):
    mean_0 = data[labels == 0].mean(axis=0).reshape(1, -1)
    mean_1 = data[labels == 1].mean(axis=0).reshape(1, -1)

    cosine_sim = cosine_similarity(mean_0, mean_1)[0][0]
    print(f"Cosine similarity between class 0 and 1 mean vectors: {cosine_sim:.4f}")
    return cosine_sim

def cosine_similarity_all_pairs(X, y):
    # Split into class 0 and class 1
    X0 = X[y == 0]
    X1 = X[y == 1]

    # Compute pairwise cosine similarity (every 0 vs every 1)
    cos_sims = cosine_similarity(X0, X1)  # Shape = (num_class0, num_class1)

    # Flatten to a single list of scores
    cos_sims_flat = cos_sims.flatten()

    print(f"Mean cosine similarity across all pairs: {np.mean(cos_sims_flat):.4f}")
    print(f"Median cosine similarity: {np.median(cos_sims_flat):.4f}")
    return cos_sims_flat

def average_cosine_similarity_by_class(X, y):
    sim_matrix = cosine_similarity(X)

    n = len(y)
    same_class = []
    diff_class = []

    for i in range(n):
        for j in range(i + 1, n):  # No need to check twice (i,j) == (j,i)
            if y[i] == y[j]:
                same_class.append(sim_matrix[i, j])
            else:
                diff_class.append(sim_matrix[i, j])

    same_avg = np.mean(same_class)
    diff_avg = np.mean(diff_class)

    print(f"🟢 Average SAME-class cosine similarity: {same_avg:.4f}")
    print(f"🔴 Average DIFFERENT-class cosine similarity: {diff_avg:.4f}")

    return same_avg, diff_avg

def compute_distances_and_plot(X_train, X_val, X_test, Y_train, Y_val, Y_test, description=""):
    # Distances
    L2distance_All = {
        "L2distance_Train": returnL2Distance(X_train, Y_train),
        "L2distance_Val": returnL2Distance(X_val, Y_val),
        "L2distance_Test": returnL2Distance(X_test, Y_test)
    }
    CosineSimilarity_All = {
        "cosineSimilarity_means_Train": cosine_similarity_between_means(X_train, Y_train),
        "cosineSimilarity_means_Val": cosine_similarity_between_means(X_val, Y_val),
        "cosineSimilarity_means_Test": cosine_similarity_between_means(X_test, Y_test)
    }
    CosineSimilarity_AvgByClass_All = {
        "cosineSimilarity_avgByClass_Train": average_cosine_similarity_by_class(X_train, Y_train),
        "cosineSimilarity_avgByClass_Val": average_cosine_similarity_by_class(X_val, Y_val),
        "cosineSimilarity_avgByClass_Test": average_cosine_similarity_by_class(X_test, Y_test)
    }

    print(f" ----- {description} ----- ")
    print(L2distance_All)
    print(CosineSimilarity_All)
    print(CosineSimilarity_AvgByClass_All)

    # Plot cosine similarity between individual vectors (only training set here)
    all_cosine_scores = cosine_similarity_all_pairs(X_train, Y_train)

    return L2distance_All, CosineSimilarity_All, CosineSimilarity_AvgByClass_All, all_cosine_scores


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

def write_csv01(data, file_path_caseName, subfolderName, filenameVars, formatted_datetime=None, prefix="data", use_pandas=True, verbose=True):
    if formatted_datetime is None:
        formatted_datetime = returnFormattedDateTimeNow()

    filename = f"{file_path_caseName}_{prefix}{filenameVars}{formatted_datetime}.csv"
    filenameFull = returnFilepathToSubfolder(filename, subfolderName)

    start_time = time.time()
    if use_pandas:
        pd.DataFrame(data).to_csv(filenameFull, index=False, header=False)
    else:
        with open(filenameFull, "w", newline="") as f:
            csv.writer(f).writerows(data)
    # I noticed how csv is faster than pandas (e.g. 0.93 vs 12.98 seconds), because pandas fills up the file with commas, while csv does not

    if verbose:
        print(f"✅ CSV ({prefix}) written in {time.time() - start_time:.2f} seconds: {filenameFull}")

def createResultsFile01(config, metadata_ALL, labels, total_timeseries_time, durationStats_Dict, samplerateStats_Dict, file_path_caseName,
                      subfolderName, filenameVars, formatted_datetime=None, prefix="result"):
    if formatted_datetime is None:
        formatted_datetime = returnFormattedDateTimeNow()

    filename = f"{file_path_caseName}_{prefix}{filenameVars}{formatted_datetime}.csv"
    filenameFull = returnFilepathToSubfolder(filename, subfolderName)

    # Save everything into a single CSV file
    with open(filenameFull, "w") as f:
        f.write(f"labels shape: {len(labels)}")
        f.write(f"\nTotal timeseries time: {total_timeseries_time:.2f} seconds\n")

        f.write("\nTHE METRICS FOR duration FOLLOW:\n")
        json.dump(durationStats_Dict, f, indent=4, default=str)
        f.write("\n\nTHE METRICS FOR samplerate FOLLOW:\n")
        json.dump(samplerateStats_Dict, f, indent=4, default=str)

        config_serializable = config.copy()
        f.write("\n\nTHE WHOLE CONFIG FOLLLOWS:\n")
        json.dump(config_serializable, f, indent=4)
        f.write("\n\n")

        f.write(metadata_ALL.to_csv(index=False).replace(",", ", "))

def SaveEmbeddingsToOutput02(filepath_data, embeddings, labels, subfolderName, formatted_datetime, indices=None, setType="NO", **kwargs):
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
def saveResultsFile02(config, filepath_data, filepath_labels, all_metrics, n_clusters_list, allDataShapes, allSegmenthapes,
                    skipgram_history, total_skipgram_time, epochEarlyStopped, tokens_train, subfolderName, formatted_datetime, setType="NO", **kwargs):
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

        for key, value in all_metrics.items():
            f.write(f"{key}: {value}\n")
      #  f.write("\nClusters --- Training Time --- Silhouette Scores:\n")
      #  for i in range(0, len(all_Silhouettes)):
      #     f.write(f"{n_clusters_list[i]} --- {all_KMeans_times[i]:.6f}--- {all_Silhouettes[i]:.6f} \n")
        df_metrics = pd.DataFrame(all_metrics) # Convert to DataFrame
        df_metrics.to_csv(f, index=False) # Save to CSV

        config_serializable = config.copy()
        config_serializable["optimizer_skipgram"] = str(config["optimizer_skipgram"])  # Convert optimizer to string
        config_serializable["skipgram_loss"] = str(config["skipgram_loss"])  # Convert L2
        config_serializable["scaler"] = str(config.get("scaler", ""))
        f.write("\nTHE WHOLE CONFIG FOLLLOWS:")
        json.dump(config_serializable, f, indent=4)

        f.write("\nSKIPGRAM MODEL HISTORY:\n")
        df_history.to_csv(f, index=False)

        f.write(f"Total Skipgram time: {total_skipgram_time:.2f} seconds\n")
        f.write(f"Epoch early stopped: {epochEarlyStopped}\n")

        returnDistribution(tokens_train, "Token", file=f)

    return

def save_data_to_csv03(data, labels, subfolderName, suffix, data_type):
    # Helper function to save both data and labels to CSV
    data_df = pd.DataFrame(data)
    labels_df = pd.DataFrame(labels)

    data_filename = f"Data_{data_type}_{suffix}.csv"
    labels_filename = f"Labels_{data_type}_{suffix}.csv"

    data_filepath = returnFilepathToSubfolder(data_filename, subfolderName)
    labels_filepath = returnFilepathToSubfolder(labels_filename, subfolderName)

    data_df.to_csv(data_filepath, index=False, header=False)
    labels_df.to_csv(labels_filepath, index=False, header=False)

def saveTrainingMetricsToFile03(config, history, model, learning_rate, optimizer, training_time, test_metrics, filepathsAll, predictions, actual_labels,
                              filepath_data, fileNameParams, ratio_0_to_1_ALL, cosineMetrics, cm_raw, cm_norm):
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

        f.write("\n\nL2distance_Means_All")
        json.dump(cosineMetrics["L2distance_Means_All"], f, indent=4)

        f.write("\n\nCosineSimilarity_Means_All")
        json.dump(cosineMetrics["CosineSimilarity_Means_All"], f, indent=4)

        f.write("\n\nCosineSimilarity_AvgByClass_All")
        json.dump(cosineMetrics["CosineSimilarity_AvgByClass_All"], f, indent=4)

        f.write("\n\nall_cosine_scores")
        all_cosine_scores = pd.DataFrame(cosineMetrics["all_cosine_scores"]).transpose()
        all_cosine_scores.to_csv(f, index=False)

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

        f.write("\nCONFUSION MATRIX FOLLOWS:\n")
        df_conf = pd.DataFrame(cm_raw)
        df_conf.to_csv(f, index=False); f.write("\n")
        df_conf = pd.DataFrame(cm_norm)
        df_conf.to_csv(f, index=False, header=True, float_format="%.3f"); f.write("\n")

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

import json
import os

import librosa
import soundfile as sf
from datetime import datetime
import random
import numpy as np
import pandas as pd
from pandas._testing import iloc
from sklearn.model_selection import train_test_split


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

def saveTrainingMetricsToFile(history, model, config, learning_rate, optimizer, training_time, test_metrics, filepathsAll, predictions, actual_labels,
                              filepath_data, fileNameParams, ratio_0_to_1_ALL, L2distance_All, cm_raw, cm_norm):
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

        f.write("\n\nL2 distances all")
        json.dump(L2distance_All, f, indent=4)

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

def analyze_audio(file_path, target_sr=44100):
    # Load audio (converts to mono by default)
    y, sr = librosa.load(file_path, sr=target_sr)

    # Duration in seconds
    duration = librosa.get_duration(y=y, sr=sr)

    # Perform FFT to get frequency components
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)

    # Threshold to ignore noise (e.g., 1% of max)
    threshold = 0.01 * np.max(fft)
    max_freq = freqs[fft > threshold].max() if any(fft > threshold) else 0

    return sr, duration, max_freq
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

def returnL2Distance(data, labels):
    mean_0 = data[labels == 0].mean(axis=0)
    mean_1 = data[labels == 1].mean(axis=0)

    dist = np.linalg.norm(mean_0 - mean_1)
    print(f"L2 distance between class 0 and 1 mean vectors: {dist:.4f}")
    return dist
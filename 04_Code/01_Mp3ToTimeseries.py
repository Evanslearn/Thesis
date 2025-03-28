import os
from os import walk
from os.path import isfile, join
import csv
import pandas as pd
import time
from datetime import datetime
import librosa
import numpy as np
from utils00 import returnFilepathToSubfolder

def loadMp3AndConvertToTimeseries(file_path, sr=None, printFlag = "No"):
    # Load MP3 with file_path
    data, sample_rate = librosa.load(file_path, sr=sr)  # sr=None keeps original sampling rate, default sr=22050

    if printFlag != "No":
        # Print basic info
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Audio Length: {len(data) / sample_rate} seconds")
        print(f"Time Series Shape: {data.shape}")

    return data, sample_rate

def extract_speaker_segments(audio, frame_length=2048, hop_length=512, threshold = 0.02):
    """
    Detects speaker activity (speech activity).
    """
    # Use RMS energy to detect speech regions
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).flatten()

    # Filter based on some threshold for speech
    speech_segments = rms > threshold  # Boolean array for active speech regions
    speech_detected = np.any(speech_segments)  # Check if any segment has speech

    return speech_segments, rms, speech_detected

def extract_time_series_from_conversation(mp3_path, sample_rate=22050, frame_length=2048, hop_length=512, threshold = 0.02):
    """
    Extracts a time series of features from a conversation.
    """
    audio, sr = loadMp3AndConvertToTimeseries(mp3_path, sr=sample_rate)

    # Extract speech regions
    speech_segments, rms, speech_detected = extract_speaker_segments(audio, frame_length, hop_length, threshold)

    # Create time series based on RMS energy
    time_series = rms[speech_segments]  # Filter only active regions

    # Return time series and energy
    return time_series, speech_detected

def preprocess_time_series(time_series, timeseries_length=512):
    """
    Preprocess a time series for normalization and resampling.
    """
    x_original = np.linspace(0, 1, len(time_series))
    x_new = np.linspace(0, 1, timeseries_length)
    time_series_resampled = np.interp(x_new, x_original, time_series)
  #  print(f"a time series {time_series.shape}")

    # Normalize - This is per-sample normalization, so it's fine (no data leakage from one sample to another - so no train-test data leakage)
    eps = 1e-8
    time_series_normalized = (time_series_resampled - np.min(time_series_resampled)) / (
            np.max(time_series_resampled) - np.min(time_series_resampled) + eps
    )
    # if i want to try z-score norm
 #   time_series_normalized = (time_series_resampled - np.mean(time_series_resampled)) / (
 #               np.std(time_series_resampled) + eps)

    return time_series_normalized


def pad_or_truncate(audio, target_length):
    if len(audio) > target_length:
        return audio[:target_length]  # Truncate if too long
    return np.pad(audio, (0, target_length - len(audio)))  # Pad if too short


def createFileLabels(labels, subfolderName, filenameVars, formatted_datetime=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):

    df_labels = pd.DataFrame(labels)
    filename = file_path_specific + f"_labels" + filenameVars + formatted_datetime + f".csv"
    filenameFull = returnFilepathToSubfolder(filename, subfolderName)
    df_labels.to_csv(filenameFull, index=False, header=False)

    return

def createFileCsv_pandas(padded_data, subfolderName, formatted_datetime=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
    start_time = time.time()

    # After all files have been processed, convert to DataFrame and write to CSV
    df = pd.DataFrame(padded_data)
    #     df = pd.DataFrame(csv_data, columns=["label", "sampling rate", "length", "FEATURES i to N"])
    filename = file_path_specific + "_output_" + "sR" + str(sample_rate) + "_" + formatted_datetime + ".csv"
    filenameFull = returnFilepathToSubfolder(filename, subfolderName)

    # Writing to CSV with pandas (which is generally faster)
    df.to_csv(filenameFull, index=False, header=False)

    pandas_writer_time = time.time() - start_time
    print(f"Pandas Time: {pandas_writer_time:.2f} seconds")
    print(f"Data written to csv file - {filenameFull}")

def createFileCsv_simple(padded_data, subfolderName, filenameVars, formatted_datetime=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
    filename = file_path_specific + "_output" + filenameVars + formatted_datetime  + ".csv"
    filenameFull = returnFilepathToSubfolder(filename, subfolderName)
    start_time = time.time()

    # After all files have been processed, write the data to a CSV file
    with open(filenameFull, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        #        writer.writerow(["label", "sampling rate", "length", "FEATURES i to N"])

        # Write the rows from processed_data into the CSV file
        writer.writerows(padded_data)

    csv_writer_time = time.time() - start_time
    print(f"CSV Writer Time: {csv_writer_time:.2f} seconds")
    print(f"Data written to csv file - {filenameFull}")

def logicForLu():
    # Example for 1 specific file
    file_path = "G:/My Drive/00_AI Master/3 Διπλωματική/05_Data/Lu/Control/F22.mp3"
    data, sample_rate = loadMp3AndConvertToTimeseries(file_path)

#    preprocessed_data = preprocess_data(parsed_data)

    file_path_base = "G:/My Drive/00_AI Master/3 Διπλωματική/05_Data/"
    file_path_specific = "Lu"
    file_path = os.path.join(file_path_base, file_path_specific)

    print(os.listdir(file_path))
    onlyfiles = [file for file in os.listdir(file_path) if isfile(join(file_path, file))]
    print(onlyfiles)

    sr = 50

    files = []
    labels = []
    time_series_data = []  # List to store the timeseries data for each mp3
    time_series_sampleRate = []

    csv_data = []

    # Loop through all subdirectories and files in the specified directory
    for (dirpath, dirnames, filenames) in walk(file_path):
        # Check if 'Control' or 'Dementia' is part of the directory path
        if "Control" in dirpath:
            label = "C"
        elif "Dementia" in dirpath:
            label = "D"
        else:
            continue  # Skip directories that aren't labeled 'Control' or 'Dementia'

        # Process files in the current directory
        for filename in filenames:
            if filename.endswith(".mp3"):  # Check for mp3 files

                files.append(filename)
                labels.append(label)  # Append the appropriate label (C or D)

                if len(labels)%5 == 0:
                    print(f"   --- Processing file number ---   {len(labels)}")

                file_path_mp3 = join(dirpath, filename)  # Full path to the MP3 file
                data, sample_rate = loadMp3AndConvertToTimeseries(file_path_mp3, sr=sr)
                time_series_data.append(data)  # Store the result
                time_series_sampleRate.append(sample_rate)

                # Get the length of the time series data
                length = len(data)

                # Store the data in the list
                csv_data.append([label, sample_rate, length] + list(data))

    #    print(time_series_data)
#    print(time_series_sampleRate)
    # Get a list of shapes
    shapes = [arr.shape for arr in time_series_data]
    # Print the list of shapes
    print(shapes)

    # Step 2: Find maximum length
    max_length = max([len(data) for data in time_series_data])
    print(f"Maximum Length of Time Series: {max_length}")

    # Step 3: Pad/Truncate and re-store csv_data
    padded_data = []
    csv_data_padded_ALL = []
    for i, data in enumerate(time_series_data):
        padded_data.append(pad_or_truncate(data, max_length))
        csv_data_padded_ALL.append([labels[i], time_series_sampleRate[i], len(padded_data)] + list(padded_data))
    print(type(padded_data)); print(type(csv_data_padded_ALL))

    print(f"\n\n\nFile Path = file_path_base + file_path_specific")
    print(f"{file_path} = {file_path_base} + {file_path_specific}")
    print(files)
    print(labels)
    print(len(labels))
    print(f"Counts of Control = {labels.count('C')}")
    print(f"Counts of Dementia = {labels.count('D')}")

    filenameVars = f"_sR{sample_rate}_"
    subfolderName = '01_TimeSeriesData'

    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    createFileLabels(labels, subfolderName, filenameVars, formatted_datetime)

    flagPandas = "Yes"
    flagCSV = "Yes"

    if flagPandas == "Yes":
        createFileCsv_pandas(padded_data, subfolderName, formatted_datetime)
    if flagCSV == "Yes":
        createFileCsv_simple(padded_data, filenameVars, subfolderName)
        # I noticed how csv is faster than pandas (e.g. 0.93 vs 12.98 seconds), because pandas fills up the file with commas, while csv does not

def printLabelCounts(labels):
    print(f"Length of labels = {len(labels)}")
    print(f"Counts of Control = {labels.count('C')}")
    print(f"Counts of Dementia = {labels.count('D')}")

# Main Workflow
if __name__ == "__main__":

    file_path_base = os.getcwd() + "../05_Data"
    file_path_base = os.path.abspath(os.path.join(os.getcwd(), "..", "05_Data"))
    print(file_path_base)

    subfolderName = '01_TimeSeriesData'

    categories = {
        "Control": os.path.join(file_path_base, "Pitt/Control/cookie"),
        "Alzheimer": os.path.join(file_path_base, "Pitt/Dementia/cookie")
    }

    # Prepare file paths for output
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path_specific = "Pitt"
    file_path = os.path.join(file_path_base, file_path_specific)
    sample_rate = int(22050/2)  # Example sampling rate

    files = []
    labels = []
    time_series_data = []  # List to store the timeseries data for each mp3
    time_series_sampleRate = []

    csv_data = []

    # Loop through all subdirectories and files in the specified directory
    for (dirpath, dirnames, filenames) in walk(file_path):
        # Check if 'Control' or 'Dementia' is part of the directory path
        if "Control" in dirpath:
            label = "C"
        elif "Dementia" in dirpath:
            label = "D"
        else:
            continue  # Skip directories that aren't labeled 'Control' or 'Dementia'
        for filename in filenames:
            if filename.endswith(".mp3"):  # Check for mp3 files

                files.append(filename)
                labels.append(label)  # Append the appropriate label (C or D)

         #       if len(labels) % 40 == 0:                    print(f"   --- Processing file number ---   {len(labels)}")
                file_path_mp3 = join(dirpath, filename)  # Full path to the MP3 file
    printLabelCounts(labels)

    frame_length = 2048
    hop_length = 512
    threshold = 0.00

    filenameVars = f"_sR{sample_rate}_frameL{frame_length}_hopL{hop_length}_thresh{threshold}_"

    time_series_processed_ALL = []
    valid_files = []  # Track filenames that are kept
    valid_labels = []  # Store labels for the kept files
    files_without_speech = []  # Track files with no speech detected
    for category, folder in categories.items():
        mp3_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".mp3")]

        start_time = time.time()
        for idx, mp3_file in enumerate(mp3_files):
            if idx % 20 == 0:
                print(f"Row number = {idx}")
            # Extract time series from conversation
            time_series, speech_detected = extract_time_series_from_conversation(mp3_file, sample_rate, frame_length, hop_length, threshold)
    #        print(category, folder)

            if not speech_detected:
                files_without_speech.append(mp3_file)  # Log the file if no speech is detected

            # Preprocess and generate time series
            if len(time_series) > 0:
                time_series_processed_ALL.append(preprocess_time_series(time_series))
                valid_files.append(mp3_file)

                # Find the correct label based on the filename
     #           if mp3_file.startswith(folder):
                valid_labels.append("C" if category == "Control" else "D")  # Assign label correctly

    total_timeseries_time = time.time() - start_time
    print(f"Total timeseries time: {total_timeseries_time:.2f} seconds")



    # Print or save the list of problematic files
    if files_without_speech:
        print(f"Files with no detected speech: {len(files_without_speech)}")
        for file in files_without_speech:
            print(f"  - {file}")

    # Now, filter labels based on valid_files
    createFileLabels(valid_labels, subfolderName, filenameVars, formatted_datetime)
    printLabelCounts(valid_labels)

    assert len(time_series_processed_ALL) == len(valid_labels), "Mismatch between time series data and labels!"
    output_filename = os.path.join(file_path_specific, f"_sR{sample_rate}_frameL{frame_length}_hopL{hop_length}_thresh{threshold}_{formatted_datetime}_output.csv")

    createFileCsv_simple(time_series_processed_ALL, subfolderName, filenameVars)
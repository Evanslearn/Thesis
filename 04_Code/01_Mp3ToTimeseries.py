import os
from os import walk
from os.path import isfile, join
from collections import Counter
import csv
import pandas as pd
import time
import librosa
import numpy as np
from matplotlib import pyplot as plt

from utils00 import returnFilepathToSubfolder, returnFormattedDateTimeNow, returnDistribution, plot_token_distribution


def loadMp3AndConvertToTimeseries(file_path, sample_rate=None, verbose=False):
    # Load MP3 with file_path
    data, sr = librosa.load(file_path, sr=sample_rate)  # sr=None keeps original sampling rate, default sr=22050

    if verbose:
        # Print basic info
        print(f"Sample Rate: {sr} Hz")
        print(f"Audio Length: {len(data) / sr} seconds")
        print(f"Time Series Shape: {data.shape}")

    return data, sr

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
    audio, sr = loadMp3AndConvertToTimeseries(mp3_path, sample_rate=sample_rate)

    # Extract speech regions
    speech_segments, rms, speech_detected = extract_speaker_segments(audio, frame_length, hop_length, threshold)

    # Create time series based on RMS energy
    time_series = rms[speech_segments]  # Filter only active regions

    # Return time series and energy
    return time_series, speech_detected

def scale_and_resample_timeseries(time_series, timeseries_length=512):
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

def write_csv(data, file_path_caseName, subfolderName, filenameVars, formatted_datetime=None, prefix="output", use_pandas=True, verbose=True):
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

def extract_audio_features(audio, sr, n_mfcc=13, hop_length=512, verbose=False):
    if len(audio) < hop_length * 2:
        raise ValueError(f"Audio too short for hop_length={hop_length} (len={len(audio)})")

    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)

        # Sanity checks
        if mfccs.shape[1] < 2 or spectral_centroid.shape[1] < 2 or chroma.shape[1] < 2:
            raise ValueError(f"Insufficient frames: MFCC={mfccs.shape}, Centroid={spectral_centroid.shape}, Chroma={chroma.shape}")

        # Mean-pool across time axis
        mfccs_mean = np.mean(mfccs, axis=1)
        centroid_mean = np.mean(spectral_centroid, axis=1)
        chroma_mean = np.mean(chroma, axis=1)

        # Final feature vector
        features = np.concatenate([mfccs_mean, centroid_mean, chroma_mean])

        # Check for NaNs or infs
        if np.isnan(features).any() or np.isinf(features).any():
            raise ValueError("Feature vector contains NaN or Inf")

     #   if verbose:
     #       print(f"✅ Feature vector extracted: shape={features.shape}")

        return features

    except Exception as e:
        if verbose:
            print(f"❌ Feature extraction error: {type(e).__name__} - {e}")
        raise e


def extract_mfcc_timeseries(audio, sr, n_mfcc=13, target_length=40, hop_length=512, threshold=0.02):
    # Step 1: Compute RMS and MFCC
    rms = librosa.feature.rms(y=audio, hop_length=hop_length).flatten()
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # Step 2: Mask frames with low RMS (likely noise or silence)
    valid_frames = rms > threshold
    if not np.any(valid_frames):
        raise ValueError("No speech detected (all RMS below threshold).")

    # Step 3: Filter MFCC frames
    mfcc_filtered = mfcc[:, valid_frames]

    # Resample each MFCC to target length
    mfcc_resampled = np.array([
        np.interp(np.linspace(0, 1, target_length),
                  np.linspace(0, 1, mfcc_filtered.shape[1]),
                  mfcc_filtered[i]) for i in range(n_mfcc)
    ])

    # Flatten to 1D feature vector
    return mfcc_resampled.flatten()


def collect_labeled_files(file_path, valid_labels=("Control", "Dementia")):
    labels = []
    labeled_files = []
    for dirpath, _, filenames in walk(file_path):
        # Check if 'Control' or 'Dementia' is part of the directory path
        if "Control" in dirpath:
            label = "C"
        elif "Dementia" in dirpath:
            label = "D"
        else:
            continue  # Skip directories that aren't labeled 'Control' or 'Dementia'
        for f in filenames:
            if f.endswith(".mp3"):
                labels.append(label)
                labeled_files.append((join(dirpath, f), label))
    return labels, labeled_files

def printLabelCounts(labels):
    counts = Counter(labels)
    print(f"Label Counts: {dict(counts)}")

# Main Workflow
def logicForPitt():

    file_path_base = os.path.abspath(os.path.join(os.getcwd(), "..", "05_Data"))
    print(file_path_base)

    subfolderName = '01_TimeSeriesData'

    file_path_caseName = "Pitt"
    file_path = os.path.join(file_path_base, file_path_caseName)

    formatted_datetime = returnFormattedDateTimeNow()

    labels, labeled_files = collect_labeled_files(file_path)
    printLabelCounts(labels)

    sample_rate, frame_length, hop_length, threshold = (
        config["sample_rate"],
        config["frame_length"],
        config["hop_length"],
        config["threshold"]
    )

    output_timeseries_ALL = []
    valid_files = []  # Track filenames that are kept
    valid_labels = []  # Store labels for the kept files
    files_without_speech = []  # Track files with no speech detected
    metadata_ALL = []

    start_time = time.time()
    for idx, (mp3_file, label) in enumerate(labeled_files):
        if idx % 20 == 0:
            print(f"Processing file #{idx}: {mp3_file}")

        # Extract time series from conversation
    #    time_series, speech_detected = extract_time_series_from_conversation(mp3_file, sample_rate, frame_length, hop_length, threshold)

  #      if not speech_detected:
   #         files_without_speech.append(mp3_file)  # Log the file if no speech is detected

        # Preprocess and generate time series
#        if len(time_series) > 0:
 #           time_series_processed_ALL.append(scale_and_resample_timeseries(time_series))
#            valid_files.append(mp3_file)


        try:
            audio, sr = loadMp3AndConvertToTimeseries(mp3_file, sample_rate=sample_rate)
            feature_type = config["feature_type"]
            resample = config["resampleTimeseries"]

            if feature_type == "raw":
                features = audio  # raw waveform
            elif feature_type == "mfcc":
                features = extract_mfcc_timeseries(audio, sr, n_mfcc=13, target_length=40, hop_length=hop_length, threshold=threshold)
            elif feature_type == "audio_features":
                features = extract_audio_features(audio, sr, hop_length=hop_length, verbose=False)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            # Optional resampling (for raw or feature vectors)
            if resample:
                output_timeseries = scale_and_resample_timeseries(features)
            else:
                output_timeseries = features  # keep original

            # Append outputs
            output_timeseries_ALL.append(output_timeseries)
            valid_labels.append(label)
            valid_files.append(mp3_file)

            # ✅ STEP 4: Logging feature shape
            if config.get("verbose"):
                duration_sec = len(audio) / sr
                shape = output_timeseries.shape
                metadata_ALL.append((os.path.basename(mp3_file), duration_sec, shape, label))
                print(f"✅ {os.path.basename(mp3_file)} | Duration: {duration_sec:.2f}s | Feature shape: {output_timeseries.shape} | Label: {label}")

        except Exception as e:
            print(f"❌ Error processing file {mp3_file}: {e}")
            continue

    total_timeseries_time = time.time() - start_time
    print(f"Total timeseries time: {total_timeseries_time:.2f} seconds")

    # Print or save the list of problematic files
    if files_without_speech:
        print(f"Files with no detected speech: {len(files_without_speech)}")
        for file in files_without_speech:
            print(f"  - {file}")

    # Now, filter labels based on valid_files
    assert len(output_timeseries_ALL) == len(valid_labels), "Mismatch between time series data and labels!"

    df_metadata = pd.DataFrame(metadata_ALL, columns=["filename", "duration", "shape", "label"])
    print(df_metadata.head())

    labels = ["ALL", "C", "D"]
  #  fig, axs = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), constrained_layout=True)
    fig, axs = plt.subplots(len(labels), 1, figsize=(8, 5 * len(labels)), constrained_layout=True)

    for i, label in enumerate(labels):
        if label == "ALL":
            subset = df_metadata
        else:
            subset = df_metadata[df_metadata["label"] == label]

        if not subset.empty:
            print(f"\nLabel: {label}")
            durations, counts = returnDistribution(subset["duration"], f"Duration for Label {label}", display=False)

            duration_mean = subset["duration"].mean()
            duration_std = subset["duration"].std()
            duration_count = subset["duration"].count()

            print(f"  Duration Mean: {duration_mean:.2f}s, Std: {duration_std:.2f}s, Count: {duration_count}")

            # Plot using the updated function, passing the correct axes
            plot_token_distribution(
                data=subset["duration"],
                name="Duration (s)",
                bins=50,
                title=f"{label} Labels",
                stats={
                    "mean": duration_mean,
                    "std": duration_std,
                    "count": duration_count
                },
                ax=axs[i]
            )
    # After plotting everything, unify the X-axis range
    common_xlim = (0, df_metadata["duration"].max())
    for ax in axs:
        ax.set_xlim(common_xlim)

    plt.suptitle("Duration Distributions by Label", fontsize=18, weight='bold')
    plt.show()

    filenameVars = f"_{feature_type}_sR{sample_rate}"

    # Only include if they are used in this feature type
    if feature_type in ["mfcc", "audio_features"]:
        filenameVars += f"_hopL{hop_length}"

    if feature_type in ["mfcc"]:
        filenameVars += f"_thresh{threshold}"

    if feature_type in ["audio_features", "raw"]:
        filenameVars += f"_frameL{frame_length}"

    filenameVars += "_"
   # filenameVars = f"_{feature_type}_sR{sample_rate}_frameL{frame_length}_hopL{hop_length}_thresh{threshold}_"
 #   printLabelCounts(valid_labels)
    write_csv(valid_labels, file_path_caseName, subfolderName, filenameVars, formatted_datetime, prefix="labels", use_pandas=True)

    #   output_filename = os.path.join(file_path_caseName, f"_sR{sr}_frameL{frame_length}_hopL{hop_length}_thresh{threshold}_{formatted_datetime}_output.csv")

    write_csv(output_timeseries_ALL, file_path_caseName, subfolderName, filenameVars, formatted_datetime, prefix="output", use_pandas=False)

config = {
    "sample_rate": None, #int(600 / 2), # None, int(22050 / 2)
    "frame_length": 2048,
    "hop_length": 512,
    "threshold": 0.02,
    "feature_type": "raw",  # Options: "raw", "mfcc", "audio_features"
    "resampleTimeseries": False,
    "verbose": True
}

logicForPitt()
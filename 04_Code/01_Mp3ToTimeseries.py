import glob
import os
from os.path import join
import pandas as pd
import time
import librosa
import numpy as np
import soundfile as sf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils00 import returnFormattedDateTimeNow, returnDistribution, printLabelCounts, write_output01, createResultsFile01, \
    read_file_returnPadded_And_lengths, readCsvAsDataframe, returnDataAndLabelsWithoutNA, makeLabelsInt, print_data_info
from utils_Plots import plot_colName_distributions, plot_audio_feature


def loadMp3AndConvertToTimeseries(file_path, sample_rate=None, verbose=False):
    # Load MP3 with file_path
    data, sr = librosa.load(file_path, sr=sample_rate)  # sr=None keeps original sampling rate, default sr=22050

    if verbose:
        # Print basic info
        print(f"Sample Rate: {sr} Hz")
        print(f"Audio Length: {len(data) / sr} seconds")
        print(f"Time Series Shape: {data.shape}")

    return data, sr

def analyze_audio(file_path, threshold, target_sr=44100):
    # Load audio (converts to mono by default)
    y, sr = librosa.load(file_path, sr=target_sr)

    # Duration in seconds
    duration = librosa.get_duration(y=y, sr=sr)

    # Perform FFT to get frequency components
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)

    # Threshold to ignore noise (e.g., 1% of max)
    thresholdFFT = threshold * np.max(fft)
    max_freq = freqs[fft > thresholdFFT].max() if any(fft > thresholdFFT) else 0

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

def returnRMSClippedSignal(signalToClip, frame_length, hop_length, threshold=0.00, is_mfcc=False, originalAudio=None):
    if is_mfcc:
        if originalAudio is None:
            raise ValueError("Must provide original waveform to calculate RMS for MFCCs!")
        rms = librosa.feature.rms(y=originalAudio, frame_length=frame_length, hop_length=hop_length).flatten()
        valid_mask = rms > threshold
        if not np.any(valid_mask):
            raise ValueError("No valid frames after RMS filtering.")
        return signalToClip[:, valid_mask]  # Keep only good frames (columns)
    else:
        rms = librosa.feature.rms(y=signalToClip, frame_length=frame_length, hop_length=hop_length).flatten()
        frames = librosa.util.frame(signalToClip, frame_length=frame_length, hop_length=hop_length)
        valid_mask = rms > threshold
        if not np.any(valid_mask):
            raise ValueError("No valid frames after RMS filtering.")
        return frames[:, valid_mask].flatten()  # Flatten back

def extract_mfcc_timeseries(audio, sr, n_mfcc, frame_length, hop_length, apply_rms_clipping=False, threshold=0.00, mfcc_summary=False, use_mfcc_deltas = False, resample=False, resample_length=40):
    # Step 1: Compute MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=frame_length)

    if apply_rms_clipping:
        mfcc = returnRMSClippedSignal(mfcc, frame_length, hop_length, threshold, is_mfcc=True, originalAudio=audio)

    if mfcc_summary:
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        if use_mfcc_deltas and mfcc.shape[1] >= 5:
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            delta_mean = np.mean(delta, axis=1)
            delta_std = np.std(delta, axis=1)

            delta2_mean = np.mean(delta2, axis=1)
            delta2_std = np.std(delta2, axis=1)

            return np.concatenate([
                mfcc_mean, mfcc_std,
                delta_mean, delta_std,
                delta2_mean, delta2_std
            ])
        else:
            return np.concatenate([mfcc_mean, mfcc_std])  # shape: (26,) for n_mfcc=13
    else:
        if use_mfcc_deltas and mfcc.shape[1] >= 5:
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc = np.concatenate([mfcc, delta, delta2], axis=0)  # Now shape = [13*3, T]

        # Flatten full sequence
        if resample:
            # Resample each MFCC to fixed time length
            mfcc = np.array([
                np.interp(np.linspace(0, 1, resample_length),
                          np.linspace(0, 1, mfcc.shape[1]),
                          mfcc[i]) for i in range(mfcc.shape[0])
            ])
        return mfcc.flatten()

def segment_audio_and_extract_mfccs(audio, sr, window_sec=2.0, hop_sec=1.0, config=None):
    assert config is not None, "You must provide the global config."

    window_length = int(window_sec * sr)
    hop_length = int(hop_sec * sr)
    segments = []

    for start in range(0, len(audio) - window_length + 1, hop_length):
        segment = audio[start:start + window_length]
        try:
            mfcc_flat = extract_mfcc_timeseries(
                segment, sr,
                n_mfcc=config["n_mfcc"],
                frame_length=config["frame_length"],
                hop_length=config["hop_length"],
                apply_rms_clipping=config["apply_rms_clipping_mfcc"],
                threshold=config["threshold"],
                mfcc_summary=config["mfcc_summary"],
                use_mfcc_deltas=config["use_mfcc_deltas"],
                resample=config["resampleTimeseries"],
                resample_length=config["resample_length"]
            )
            segments.append(mfcc_flat)
        except Exception as e:
            print(f"⚠️ Skipping segment due to error: {e}")
            continue

    return segments

def segment_and_extract_mfcc_means(audio, sr, config, window_sec=2.0, hop_sec=1.0):
    window_length = int(window_sec * sr)
    hop_length = int(hop_sec * sr)
    n_mfcc = config["n_mfcc"]
    features = []

    for start in range(0, len(audio) - window_length + 1, hop_length):
        segment = audio[start:start + window_length]

        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, hop_length=config["hop_length"], n_fft=config["frame_length"])

        # Mean over time axis (shape: [n_mfcc])
        mfcc_mean = np.mean(mfcc, axis=1)
        features.append(mfcc_mean)

    return np.concatenate(features)  # shape = (n_segments × n_mfcc,)



def extract_audio_features(audio, sr, n_mfcc=13, frame_length=2048, hop_length=512, verbose=False):
    if len(audio) < hop_length * 2:
        raise ValueError(f"Audio too short for hop_length={hop_length} (len={len(audio)})")

    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)

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


def collect_labeled_files(file_path, valid_labels=("Control", "Dementia")):
    labels = []
    labeled_files = []
    for dirpath, _, filenames in os.walk(file_path):
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


def count_windows_with_duration(audio, window_size, stride, sr):
    """
    Count number of windows and compute window duration.

    Args:
        audio (np.ndarray): 1D raw audio signal.
        window_size (int): Window size in samples.
        stride (int): Hop size (stride) in samples.
        sr (int): Sample rate (Hz).

    Returns:
        tuple: (number of windows, window duration in seconds, stride duration in seconds)
    """
    L = len(audio)
    n_windows = (L - window_size) // stride + 1
    window_duration_sec = window_size / sr
    stride_duration_sec = stride / sr
    return n_windows, window_duration_sec, stride_duration_sec

# Main Workflow
def logicForPitt(file_path_caseName = "Pitt"):
    file_path_base = os.path.abspath(os.path.join(os.getcwd(), "..", "05_Data"))
    print(file_path_base)

    subfolderName = '01_TimeSeriesData'
    file_path = os.path.join(file_path_base, file_path_caseName)

    formatted_datetime = returnFormattedDateTimeNow()

    labels, labeled_files = collect_labeled_files(file_path)
    printLabelCounts(labels)

    metadata_ALL = extract_duration_and_samplerate(labeled_files)
    df_metadata = pd.DataFrame(metadata_ALL, columns=["filename", "duration", "sample_rate", "label"])
    stats_duration = plot_colName_distributions(df_metadata, colName="duration", title="Duration Distributions by Label")
    stats_sampleRate = plot_colName_distributions(df_metadata, colName="sample_rate", title="Sample Rate Distributions by Label")

    sample_rate, frame_length, hop_length, threshold, apply_rms_clipping_global, apply_rms_clipping_mfcc, mfcc_summary, use_mfcc_deltas = (
        config["sample_rate"],
        config["frame_length"],
        config["hop_length"],
        config["threshold"],
        config["apply_rms_clipping_global"],
        config["apply_rms_clipping_mfcc"],
        config["mfcc_summary"],
        config["use_mfcc_deltas"]
    )
    n_mfcc = config['n_mfcc'] # 13

    output_timeseries_ALL = []
    valid_files = []  # Track filenames that are kept
    valid_labels = []  # Store labels for the kept files
    files_without_speech = []  # Track files with no speech detected
    metadata_detailed_ALL = []

    start_time = time.time()
    for idx, (mp3_file, label) in enumerate(labeled_files):
        if idx % 20 == 0:
            print(f"Processing file #{idx}: {mp3_file}")

        try:
            audio, sr = loadMp3AndConvertToTimeseries(mp3_file, sample_rate=sample_rate)

            # Analyze frequency and duration at higher precision (optional: use a higher target_sr)
            full_sr, duration, max_freq = analyze_audio(mp3_file, threshold, target_sr=None)

            feature_type = config["feature_type"]
            resample = config["resampleTimeseries"]
            resample_length = config['resample_length']

            # Clip using RMS threshold
            if apply_rms_clipping_global:
                audio = returnRMSClippedSignal(audio, frame_length=frame_length, hop_length=hop_length, threshold=threshold, is_mfcc=False)

            if feature_type == "raw":
                features = audio  # raw waveform
                # Plot raw waveform
                plot_audio_feature(audio, sr, feature_type=feature_type)
            elif feature_type == "mfcc":
                features = extract_mfcc_timeseries(audio, sr, n_mfcc=n_mfcc, frame_length=frame_length, hop_length=hop_length, apply_rms_clipping=apply_rms_clipping_mfcc, threshold=threshold,
                                                   mfcc_summary=mfcc_summary, use_mfcc_deltas=use_mfcc_deltas, resample=resample, resample_length=resample_length)
     #           features = segment_and_extract_mfcc_means(audio, sr, config, window_sec=2.0, hop_sec=1.0)
                # This is only for plotting, compute MFCC properly
    #            mfcc_for_plot = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=config["n_mfcc"], hop_length=hop_length, n_fft=frame_length)
    #            plot_audio_feature(mfcc_for_plot, sr, feature_type=feature_type, hop_length=hop_length)
            elif feature_type == "audio_features":
                features = extract_audio_features(audio, sr, frame_length=frame_length, hop_length=hop_length, verbose=False)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            # Optional resampling (for raw or feature vectors)
            if resample and not mfcc_summary:
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
                # Basic metadata
                file_metadata = [
                    os.path.basename(mp3_file),
                    sr,
                    full_sr,
                    max_freq,
                    duration_sec,
                    output_timeseries.shape,
                    label
                ]

                # Add window info IF the feature_type is one of these
                if feature_type in ["raw", "mfcc", "audio_features"]:
                    n_windows, window_duration_sec, stride_duration_sec = count_windows_with_duration(
                        audio, window_size=frame_length, stride=hop_length, sr=sr
                    )
                    file_metadata.extend([n_windows, window_duration_sec, stride_duration_sec])

                metadata_detailed_ALL.append(file_metadata)

                # Build the basic stuff
                line = (
                    f"✅ {os.path.basename(mp3_file)} | SR: {sr:.2f} | Original SR: {full_sr} | MaxF: {max_freq:.2f} Hz | "
                    f"Dur: {duration_sec:.2f}s | Shape: {output_timeseries.shape} | Label: {label}"
                )

                # Add extra for MFCC / audio_features
                if feature_type in ["mfcc", "audio_features"]:
                    line += (
                        f" | Windows: {n_windows} | WindowDur: {window_duration_sec:.4f}s | StrideDur: {stride_duration_sec:.4f}s"
                    )

                # Final print
                print(line)

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

    columns = ["filename", "SR", "OriginalSR", "max_freq", "duration", "shape", "label"]

    if feature_type in ["mfcc", "audio_features"]:
        columns.extend(["n_windows", "window_duration_sec", "stride_duration_sec"])

    df_metadata = pd.DataFrame(metadata_detailed_ALL, columns=columns)
    print(df_metadata.head())

    # df_metadata built from processed outputs
#    stats_durationApproximation = plot_colName_distributions(df_metadata, title="Processed Duration Distribution")

    filenameVars = f"_{feature_type}_sR{sample_rate}"

    if config["apply_rms_clipping_global"]:
        filenameVars += f"_thr{threshold}"
    if config["apply_rms_clipping_mfcc"]:
        filenameVars += f"_thrMFCC{threshold}"

    # Only include if they are used in this feature type
    if feature_type != "raw" or config["apply_rms_clipping_global"] or config["apply_rms_clipping_mfcc"]:
        filenameVars += f"_hopL{hop_length}"

    if feature_type == "mfcc":
        filenameVars += f"_mfcc_summary{mfcc_summary}"
        filenameVars += f"_use_mfcc_deltas{use_mfcc_deltas}"
        filenameVars += f"_nMFCC{n_mfcc}"

    if feature_type in ["mfcc", "audio_features"]:
        filenameVars += f"_nFFT{frame_length}"
    elif feature_type == "raw":
        filenameVars += f"_frameL{frame_length}"

    if config['resampleTimeseries']:
        filenameVars += f"_resample{config['resampleTimeseries']}"

    filenameVars += "_"
 #   printLabelCounts(valid_labels)
    write_output01(valid_labels, file_path_caseName, subfolderName, filenameVars, formatted_datetime, prefix="labels", use_pandas=True)
    write_output01(output_timeseries_ALL, file_path_caseName, subfolderName, filenameVars, formatted_datetime, prefix="data", file_format="npy", use_pandas=False)

    createResultsFile01(config, df_metadata, valid_labels, total_timeseries_time, stats_duration,
                      stats_sampleRate, file_path_caseName, subfolderName, filenameVars, formatted_datetime)

    stats_maxFreq = plot_colName_distributions(df_metadata, colName="max_freq", title="Max Frequency Distributions by Label")
    returnDistribution(df_metadata['OriginalSR'], "Original SR")

config = {
    "sample_rate": 44100,
    "frame_length": 2048,
    "hop_length": 1024,
    "threshold": 0.00,
    "feature_type": "mfcc",  # Options: "raw", "mfcc", "audio_features"
    "resampleTimeseries": False,
    "resample_length": 1024,
    "apply_rms_clipping_global": False,
    "apply_rms_clipping_mfcc": False,
    "mfcc_summary": False,  # True for mean+std, False for flatten
    "use_mfcc_deltas": False,  # Add Δ and ΔΔ features (summary only)
    "n_mfcc": 13, #13
    "verbose": True
}

file_path_caseName = "Pitt"
logicForPitt(file_path_caseName)



def runLocalExperiment():
    def load_features_and_labels(base_folder):
        """Helper to load your saved features and labels."""
        labels_path = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if "labels" in f and f.endswith(".csv")][-1]
        features_path = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if "data" in f and f.endswith(".csv")][-1]

        labels = pd.read_csv(labels_path)["label"].values
        features = np.loadtxt(features_path, delimiter=",")

        return features, labels

    def get_latest_data_and_label_files(folder_path):
        # Find all label and data CSVs
        label_files = glob.glob(os.path.join(folder_path, "*labels*.csv"))
        data_files = glob.glob(os.path.join(folder_path, "*data*.csv"))

        if not label_files or not data_files:
            raise ValueError("❌ No matching label or data files found!")

        # Sort by last modified time
        label_files = sorted(label_files, key=os.path.getmtime)
        data_files = sorted(data_files, key=os.path.getmtime)

        # Take the latest
        filepath_labels = os.path.basename(label_files[-1])
        filepath_data = os.path.basename(data_files[-1])

        print(f"🔵 Found LABEL file: {filepath_labels}")
        print(f"🔵 Found DATA file: {filepath_data}")

        return filepath_data, filepath_labels

    def load_latest_features_and_labels(base_folder):
        """Load the latest feature and label files based on modification time (correctly)."""
        label_files = glob.glob(os.path.join(base_folder, "*labels*.csv"))
        feature_files = glob.glob(os.path.join(base_folder, "*features.npy"))

        if not label_files or not feature_files:
            raise ValueError("❌ No matching label or feature files found!")

        label_files = sorted(label_files, key=os.path.getmtime)
        feature_files = sorted(feature_files, key=os.path.getmtime)

        latest_label_file = label_files[-1]
        latest_feature_file = feature_files[-1]

        print(f"🔵 Loading LABEL file: {latest_label_file}")
        print(f"🔵 Loading FEATURE file: {latest_feature_file}")

        labels = np.loadtxt(latest_label_file, delimiter=",", dtype=str)
        features = np.load(latest_feature_file, allow_pickle=True)

        return features, labels

    import xgboost as xgb
    import lightgbm as lgb

    def train_validate_test_pipeline(features, labels, model=None, normalization_method="standard",
                                     apply_pca=False, pca_variance_threshold=0.95):
        """Split, normalize, train, and show detailed predictions + probabilities."""
        if model is None:
            # Train XGBoost

         #   model = xgb.XGBClassifier(
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )

        # Step 1: Split
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.30, random_state=42, stratify=labels
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        if normalization_method == "standard":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            print("🔵 Applied StandardScaler normalization.")

        elif normalization_method == "minmax":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            print("🔵 Applied MinMaxScaler normalization.")

        # Step 2b: Apply PCA if requested
        if apply_pca:
            pca = PCA(n_components=pca_variance_threshold, svd_solver='full')
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            X_test = pca.transform(X_test)
            print(f"🔵 PCA applied! Reduced dimensions: {X_train.shape[1]} components.")

        # Step 3: Train
        model.fit(X_train, y_train)

        # Step 4: Predict
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        # Step 5: Predict Probabilities
        train_probs = model.predict_proba(X_train)
        val_probs = model.predict_proba(X_val)
        test_probs = model.predict_proba(X_test)

        # Step 6: Accuracy
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        test_acc = accuracy_score(y_test, test_preds)

        print(f"\n✅ Train Accuracy: {train_acc:.4f}")
        print(f"✅ Validation Accuracy: {val_acc:.4f}")
        print(f"✅ Test Accuracy: {test_acc:.4f}\n")

        # Step 7: Show Predictions
        print("🔵 Validation Predictions vs True Labels + Probabilities:")
     #   for pred, true, prob in zip(val_preds, y_val, val_probs):
    #       print(f"Predicted: {pred} | Actual: {true} | Probabilities: {np.round(prob, 3)}")

        print("\n🟢 Test Predictions vs True Labels + Probabilities:")
      #  for pred, true, prob in zip(test_preds, y_test, test_probs):
      #      print(f"Predicted: {pred} | Actual: {true} | Probabilities: {np.round(prob, 3)}")

        return train_acc, val_acc, test_acc

    # 📍 Run after logicForPitt() finishes
    # Set your folder
    timeSeriesDataPath = "/01_TimeSeriesData/"
    folderPath = os.getcwd() + timeSeriesDataPath


    # Get latest files
    filepath_data, filepath_labels = get_latest_data_and_label_files(folderPath)

    # Now you can load!
    data, lengths = read_file_returnPadded_And_lengths(os.path.join(folderPath, filepath_data))
    initial_labels = readCsvAsDataframe(folderPath, filepath_labels, dataFilename="labels", as_series=True)

    # Process
    data, labels = returnDataAndLabelsWithoutNA(data, initial_labels, addIndexColumn=True)
    labels = makeLabelsInt(labels)
    data.columns = data.columns.astype(str)

    print_data_info(data, labels, "AFTER DROPPING NA")

    # 2. Train + validate + test
    models = [LogisticRegression(
        solver='liblinear',
        random_state=42,
        max_iter=1000
    ),
    RandomForestClassifier()]

    apply_pca = True
    pca_variance_threshold = 0.95

    n_runs = 10
    for model in models:
        print(f"\nMODEL -> {model}")
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []
        for run in range(n_runs):
            print(f"🔵 Run {run + 1}/{n_runs}")
        #    val_acc, test_acc = train_validate_test_pipeline(data, labels, model=model, normalization_method="standard", apply_pca=apply_pca, pca_variance_threshold=pca_variance_threshold)
            train_acc, val_acc, test_acc = train_validate_test_pipeline(data, labels, model=model, normalization_method="minmax", apply_pca=apply_pca, pca_variance_threshold=pca_variance_threshold)

            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            test_accuracies.append(test_acc)

        # Summary
        print(f"\n✅ Final Results after {n_runs} runs:")
        print(f"Train Accuracy: {np.mean(train_accuracies):.4f} ± {np.std(train_accuracies):.4f}")
        print(f"Validation Accuracy: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
        print(f"Test Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")

    results_df = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Accuracy": [np.mean(train_accuracies), np.mean(val_accuracies), np.mean(test_accuracies)],
        "StdDev": [np.std(train_accuracies), np.std(val_accuracies), np.std(test_accuracies)]
    })
    results_df.to_csv(os.path.join(folderPath, "experiment_results.csv"), index=False)
    print("Results saved!")

#runLocalExperiment()
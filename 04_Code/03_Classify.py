import sys
import os
import time
import random
from datetime import datetime

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.svm import SVC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### TensorFlow & Keras Imports ###
import tensorflow as tf
from tensorflow import keras, sigmoid
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import MeanSquaredError, Accuracy, Precision, Recall
from keras.regularizers import l2

tf.get_logger().setLevel('ERROR')  # Suppress DEBUG logs

from imblearn.over_sampling import SMOTE, RandomOverSampler
### SciKit-Learn Imports ###
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.utils import resample
from xgboost import XGBClassifier

from utils00 import (
    returnFilepathToSubfolder,
    doTrainValTestSplit,
    plotTrainValMetrics,
    plot_bootstrap_distribution,
    saveTrainingMetricsToFile,
    makeLabelsInt, readCsvAsDataframe, plot_tsnePCAUMAP, returnFormattedDateTimeNow, returnDataAndLabelsWithoutNA,
    trim_datetime_suffix, dropInstancesUntilClassesBalance, read_padded_csv_with_lengths, return_scaler_type
)

def custom_formatter(x):
    return f"{x:.6f}"

def print_shapes(name, original, normalized):
    """Helper function to print original and normalized dataset shapes."""
    print(f'{name} shape is = {original.shape}')
    print(f'{name} normalized shape is = {normalized.shape}')

def save_data_to_csv(data, labels, subfolderName, suffix, data_type):
    # Helper function to save both data and labels to CSV
    data_df = pd.DataFrame(data)
    labels_df = pd.DataFrame(labels)

    data_filename = f"Data_{data_type}_{suffix}.csv"
    labels_filename = f"Labels_{data_type}_{suffix}.csv"

    data_filepath = returnFilepathToSubfolder(data_filename, subfolderName)
    labels_filepath = returnFilepathToSubfolder(labels_filename, subfolderName)

    data_df.to_csv(data_filepath, index=False, header=False)
    labels_df.to_csv(labels_filepath, index=False, header=False)

def calculate_class_ratios(labels_list, names_list):
    """Calculate and print the ratio of 0s to 1s for each dataset."""
    ratios = []
    for labels, name in zip(labels_list, names_list):
        count_0 = np.sum(labels == 0)
        count_1 = np.sum(labels == 1)
        ratio = count_0 / count_1
        print(f"{name}, Y_0/Y_1 = {ratio}")
        ratios.append(ratio)
    return ratios


### Global Configuration Dictionary ###
CONFIG = {
    "split_options": ["NO"], # "YES", "NO"
    "batch_size": 128,
    "epochs": 50,
    "loss": "binary_crossentropy",
    "metrics": ['accuracy', Precision(), Recall()], # metrics = ['mse', 'mae', 'accuracy'],
    "enable_scaling": True,
    "scaler": MinMaxScaler(), # None, MinMaxScaler(), StandardScaler()
    "optimizer": SGD,
    "momentum": 0.9, # for SGD
    "units": {
        "SimpleRNN": [64, 32, 32], # [32, 32]
        "GRU": [32], # 32
        "LSTM": [32, 32, 32],  # 32
    },
    "neurons": {
        "Dense": [128, 32, 64], # 64
    },
    "layers": {
        "SimpleRNN": 0, # 2
        "GRU": 0, # 0
        "LSTM": 0, # 1
        "Dense": 2, # 1ty
        "Dropout": 1,
        "BatchNorm": 1
    },
    "recurrent_dropout": 0.0, # 0.2
    "activation_dense": "relu",
    "dropout": 0.4,
    "kernel_regularizer_dense": l2(0.001) # None, l2(0.001)
}

lr_min = 0.0001 # 0.001, 0.035
lr_max = 0.1 # 0.01
lr_distinct = 10 # 10
learning_rate = np.linspace(lr_min, lr_max, num=lr_distinct).tolist()


#embeddingsPath = "/02_Embeddings/"
#folderPath = os.getcwd() + embeddingsPath

#filepath_data = "Embeddings__Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings50_2025-03-22_00-00-15.csv"
#filepath_labels = "Labels__Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings50_2025-03-22_00-00-15.csv"
filepath_data = "Pitt_output_raw_sR300_frameL2048_hopL512_thresh0.02_2025-04-08_00-11-56.csv"
filepath_labels = "Pitt_labels_raw_sR300_frameL2048_hopL512_thresh0.02_2025-04-08_00-11-56.csv"

if CONFIG['split_options']:
    import Help_03_Paths as fp
else:
    # Dummy placeholder so `fp` still exists without valid paths
    class DummyFP:
        pass
    fp = DummyFP()
if not CONFIG['split_options']:
    # Clear any accidental usage of paths
    fp.FILEPATH_DATA = None
    fp.FILEPATH_LABELS = None
    fp.FILEPATH_DATA_TRAIN = None
    fp.FILEPATH_DATA_VAL = None
    fp.FILEPATH_DATA_TEST = None
    fp.FILEPATH_LABELS_TRAIN = None
    fp.FILEPATH_LABELS_VAL = None
    fp.FILEPATH_LABELS_TEST = None
    fp.FILEPATH_INDICES = None
    fp.FILEPATH_INDICES_TRAIN = None
    fp.FILEPATH_INDICES_VAL = None
    fp.FILEPATH_INDICES_TEST = None
    fp.FOLDER_PATH = os.getcwd() + "/01_TimeSeriesData/"  # Raw folder fallback


# When without Signal2Vec
def whenWithoutSignal2Vec(filepath_data, filepath_labels):
    fp.FILEPATH_DATA = filepath_data; fp.FILEPATH_LABELS = filepath_labels
 #   data = readCsvAsDataframe(os.getcwd() + "/01_TimeSeriesData/", filepath_data, "data")
    labels = readCsvAsDataframe(os.getcwd() + "/01_TimeSeriesData/", filepath_labels, "labels", as_series=True)

    folderPath = os.getcwd() + "/01_TimeSeriesData/"
    data, lengths = read_padded_csv_with_lengths(os.path.join(folderPath, filepath_data))

    print(f"BEFORE DROPNA + PADDING -> data.shape {data.shape}")
    # Clean first
 #   data, labels = returnDataAndLabelsWithoutNA(data, labels)
    # Pad AFTER filtering
 #   data, _ = pad_variable_length_timeseries(data)

    data, labels = returnDataAndLabelsWithoutNA(data, labels)
    print(f"AFTER DROPNA + PADDING -> data.shape {data.shape}")

    labels = makeLabelsInt(labels)
    return data, labels, fp.FILEPATH_DATA, fp.FILEPATH_LABELS
data, labels, fp.FILEPATH_DATA, fp.FILEPATH_LABELS = whenWithoutSignal2Vec(filepath_data, filepath_labels)

val_ratio = 1

indices_step02_train = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_INDICES_TRAIN, "indicesTrain")
indices_step02_val = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_INDICES_VAL, "indicesVal")
indices_step02_test = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_INDICES_TEST, "indicesTest")
indices_step02 = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_INDICES, "indicesAll").to_numpy()

def returnDatasplit(needSplitting = "NO"):
    global val_ratio
    print(f"\n----- NEEDS SPLITTING == {needSplitting} -----")
    if needSplitting == "NO":
        fp.FILEPATH_DATA = fp.FILEPATH_DATA_TRAIN.replace("_trainSet", "")
        print(f"fp.FOLDER_PATH, fp.FILEPATH_DATA = {fp.FOLDER_PATH}, {fp.FILEPATH_DATA}")
        print(f"fp.FOLDER_PATH, fp.FILEPATH_DATA_TRAIN = {fp.FOLDER_PATH}, {fp.FILEPATH_DATA_TRAIN}")
        X_train = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_DATA_TRAIN, "X_train").to_numpy()
        X_val = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_DATA_VAL, "X_val").to_numpy()
        X_test = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_DATA_TEST, "X_test").to_numpy()
        Y_train = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_LABELS_TRAIN, "Y_train", as_series=True).to_numpy()
        Y_val = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_LABELS_VAL, "Y_val", as_series=True).to_numpy()
        Y_test = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_LABELS_TEST, "Y_test", as_series=True).to_numpy()
        indices_train = indices_step02_train.to_numpy(); indices_val = indices_step02_val.to_numpy(); indices_test = indices_step02_test.to_numpy()

    else:
        X_data, Y_targets = np.array(data), np.array(labels);
        print(f'\nLength of X is = {len(X_data)}. Length of Y is = {len(Y_targets)}')

        # ----- NAKE COUNT OF 0s AND 1s BE THE SAME -----
     #   X_data, Y_targets = dropInstancesUntilClassesBalance(X_data, Y_targets)

        X_train, X_val, X_test, Y_train, Y_val, Y_test, val_ratio, indices_train, indices_val, indices_test = doTrainValTestSplit(X_data, Y_targets)

        caseTypeStrings = ["Train", "Val", "Test"]
        indicesStrings = [indices_train, indices_val, indices_test]
        formatted_datetime = returnFormattedDateTimeNow()
        subfolderName = "03_ClassificationResults"
        for i in range (0, len(caseTypeStrings)):
            df_indices = pd.DataFrame({'Indices': indicesStrings[i]})
            filename = "Indices" + caseTypeStrings[i] + "_" + formatted_datetime + ".csv"
            filenameFull = returnFilepathToSubfolder(filename, subfolderName)
            df_indices.to_csv(filenameFull, index=False, header=False)

        # SHOULD BE EMPTY ARRAYS
        print("Train/Val overlap:", np.intersect1d(indices_train, indices_val))
        print("Train/Test overlap:", np.intersect1d(indices_train, indices_test))
        print("Val/Test overlap:", np.intersect1d(indices_val, indices_test))

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, val_ratio, indices_train, indices_val, indices_test


def add_rnn_layers(model, rnn_type, num_layers, units, next_rnn_exists, recurrent_dropout=None):
    rnn_layer = {"SimpleRNN": tf.keras.layers.SimpleRNN, "GRU": tf.keras.layers.GRU, "LSTM": tf.keras.layers.LSTM}.get(rnn_type)
    if not rnn_layer or num_layers <= 0:
        return  # No layers to add

    use_recurrent_dropout = recurrent_dropout if recurrent_dropout is not None and rnn_type in ["GRU", "LSTM"] else 0.0     # Check if recurrent_dropout is specified and valid (only for GRU/LSTM)

    # Ensure `units` is properly indexed
    if isinstance(units, list):
        if len(units) < num_layers:
            raise ValueError(f"Expected at least {num_layers} units, but got only {len(units)}.")
        units_to_use = units[:num_layers]  # Use only the required number of units
    else:
        units_to_use = [units] * num_layers  # Convert single unit value into a list

    for i in range(num_layers - 1):
        model.add(rnn_layer(units_to_use[i] if isinstance(units, list) else units, return_sequences=True, recurrent_dropout=use_recurrent_dropout))

    # The last layer should return sequences only if another RNN follows
    model.add(rnn_layer(units_to_use[-1] if isinstance(units, list) else units, return_sequences=next_rnn_exists, recurrent_dropout=use_recurrent_dropout))

def check_indicesEqual(indices_step02, indices_all):
    print(f'Indices shape: step2 = {indices_step02.shape}, step3 = {indices_all.shape}')
    if not np.array_equal(indices_step02, indices_all):
        print(f"Condition failed: {indices_step02[0]} != {indices_all[0]}")
        sys.exit()  # Terminate the execution

def calculate_f1(y_true, y_pred):
    # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    precision.update_state(y_true, y_pred)
    recall.update_state(y_true, y_pred)
    return 2 * (precision.result() * recall.result()) / (precision.result() + recall.result() + tf.keras.backend.epsilon())

def model03_VangRNN(data, labels, needSplitting, config):
    # Extract hyperparameters
    batch_size, epochs = config["batch_size"], config["epochs"]
    loss, metrics = config["loss"], config["metrics"]
    learning_rate, optimizer_class  = config["learning_rate"], config["optimizer"]
    momentum = config["momentum"]

    # Extract RNN layers
    layers = config["layers"]
    SIMPLE_layers = layers.get("SimpleRNN", 0)
    GRU_layers = layers.get("GRU", 0)
    LSTM_layers = layers.get("LSTM", 0)

    # Extract RNN units and Dense neurons
    units_simple = config["units"].get("SimpleRNN", [])
    units_gru = config["units"].get("GRU", [])
    units_lstm = config["units"].get("LSTM", [])
    dense_neurons = config["neurons"]["Dense"]

    recurrent_dropout = config["recurrent_dropout"]
    dropout_rate = config["dropout"]
    activation_dense = config["activation_dense"]
    kernel_regularizer_dense = config["kernel_regularizer_dense"]

    X_train, X_val, X_test, Y_train, Y_val, Y_test, val_ratio, indices_train, indices_val, indices_test = returnDatasplit(needSplitting)
    print(Y_train); print(Y_val); print(Y_test)

    indices_all = np.vstack([indices_train.reshape(-1, 1), indices_val.reshape(-1, 1), indices_test.reshape(-1, 1)])
  #  check_indicesEqual(indices_step02, indices_all)

  #  data, labels = dropInstancesUntilClassesBalance(data, labels)

    filepathsAll = {
        "fp.FILEPATH_DATA": fp.FILEPATH_DATA,
        "fp.FILEPATH_DATA_TRAIN": fp.FILEPATH_DATA_TRAIN,
        "fp.FILEPATH_DATA_VAL": fp.FILEPATH_DATA_VAL,
        "fp.FILEPATH_DATA_TEST": fp.FILEPATH_DATA_TEST,
        "fp.FILEPATH_LABELS": fp.FILEPATH_LABELS,
        "fp.FILEPATH_LABELS_TRAIN": fp.FILEPATH_LABELS_TRAIN,
        "fp.FILEPATH_LABELS_VAL": fp.FILEPATH_LABELS_VAL,
        "fp.FILEPATH_LABELS_TEST": fp.FILEPATH_LABELS_TEST
    }

    # save data to a CSV
    subfolderName = "03_ClassificationResults"
    suffix = f"Splitting_{needSplitting}"

    # Save training, validation, and test data using the helper function
    for dataset, label in zip([X_train, X_val, X_test], ["train", "val", "test"]):
        save_data_to_csv(dataset, eval(f"Y_{label}"), subfolderName, suffix, label)

    # Add callbacks for better control
    early_stop = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.001, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)


    def tryThisTomorrow(X_train, X_val, X_test):

   #     plot_tsnePCAUMAP(PCA, X_train, Y_train, 10, 42, "of Signal2Vec Embeddings", remove_outliers=False)
   #     plot_tsnePCAUMAP(TSNE, X_train, Y_train, 10, 42, "of Signal2Vec Embeddings", remove_outliers=False)
    #    plot_tsnePCAUMAP(umap.UMAP, X_train, Y_train, 10, 42, "of Signal2Vec Embeddings", remove_outliers=False)

        mean_0 = X_train[Y_train == 0].mean(axis=0)
        mean_1 = X_train[Y_train == 1].mean(axis=0)

        dist = np.linalg.norm(mean_0 - mean_1)
        print(f"L2 distance between class 0 and 1 mean vectors: {dist:.4f}")



        # Build a minimal model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.summary()

        # Compile with a lower learning rate
        if optimizer_class == SGD:
            optimizer = optimizer_class(learning_rate=learning_rate, momentum=momentum)
        else:
            optimizer = optimizer_class(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
      #  model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        # Add callbacks for better control
        early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

        # Train
   #     history = model.fit(
   #         X_train, Y_train,
   #         validation_data=(X_val, Y_val),
  #          epochs=60, batch_size=128,
   #         callbacks=[early_stop, lr_scheduler]
   #     )

   #     predictions = model.predict(X_test)
    #    loss_evaluate = model.evaluate(X_test, Y_test)
    #    print(loss_evaluate)


        # Dictionary of models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM (RBF Kernel)": SVC(kernel='rbf'),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            # use_label_encoder=False suppresses warning
        }

        for name, clf in models.items():
            print(f"\n🔍 Training: {name}")
            clf.fit(X_train, Y_train)

            # ---- Validation evaluation ----
            val_preds = clf.predict(X_val)
            print(f"\n📊 Validation Performance for {name}:")
            print(classification_report(Y_val, val_preds))
            val_acc = accuracy_score(Y_val, val_preds)
            print(f"Validation Accuracy: {val_acc:.4f}")
            print("Validation Confusion Matrix:")
            print(confusion_matrix(Y_val, val_preds))

            # ---- Test evaluation ----
            test_preds = clf.predict(X_test)
            print(f"\n📊 Test Performance for {name}:")
            print(classification_report(Y_test, test_preds))
            test_acc = accuracy_score(Y_test, test_preds)
            print(f"Test Accuracy: {test_acc:.4f}")
            print("Test Confusion Matrix:")
            print(confusion_matrix(Y_test, test_preds))



            # Predict both sets
            val_preds = clf.predict(X_val)
            test_preds = clf.predict(X_test)

            # Get reports as dicts
            val_report = classification_report(Y_val, val_preds, output_dict=True)
            test_report = classification_report(Y_test, test_preds, output_dict=True)

            # Convert to DataFrames
            val_df = pd.DataFrame(val_report).transpose()
            test_df = pd.DataFrame(test_report).transpose()

            # Add a column to indicate the split
            val_df["set"] = "validation"
            test_df["set"] = "test"

            # Combine them
            combined_df = pd.concat([val_df, test_df], axis=0)

            # Optional: reorder for clarity
            combined_df = combined_df[["set", "precision", "recall", "f1-score", "support"]]

            print(f"\n📋 Combined classification report for {name}:")
            print(combined_df.round(2))

            np.set_printoptions(formatter={'float': custom_formatter}, linewidth=np.inf)
            print(f'Predictions shape = {test_preds.shape}');
            print(f'Y_test_normalized shape = {Y_test.shape}')

            if hasattr(clf, "predict_proba"):
                y_proba_test = clf.predict_proba(X_test)[:, 1]  # probability of class 1
            elif hasattr(clf, "decision_function"):  # e.g., SVM
                y_scores = clf.decision_function(X_test)
                y_proba_test = sigmoid(y_scores)  # You can define a sigmoid if you want probs
            else:
                y_proba_test = clf.predict(X_test)  # fallback, hard labels
            if isinstance(y_proba_test, tf.Tensor):
                y_proba_test = y_proba_test.numpy()
            print(f'pred probab = {y_proba_test.T}')
            print(f'predictions = {test_preds.T}')
            print(f'real labels = {Y_test}')

    # Normalize the data
    scaler = config["scaler"]
    scalerName = return_scaler_type(str(config.get("scaler", "")), config['enable_scaling'])
    def scaleData(scaler, data, enable_scaling=False, fit=False):
        if not enable_scaling:
            return data

        if fit:
            scaled_values = scaler.fit_transform(data)
        else:
            scaled_values = scaler.transform(data)

        return scaled_values

    print(f"X_train_normalized.shape before scaling = {X_train.shape}")
    if X_train.ndim == 3:
        X_train_normalized = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_normalized = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_normalized = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    else:
        X_train_normalized = scaleData(config["scaler"], X_train, enable_scaling=config["enable_scaling"], fit=True)
        X_val_normalized = scaleData(config["scaler"], X_val, enable_scaling=config["enable_scaling"])
        X_test_normalized = scaleData(config["scaler"], X_test, enable_scaling=config["enable_scaling"])
    print(f"X_train_normalized.shape after scaling = {X_train_normalized.shape}")

    ratio_0_to_1_ALL = calculate_class_ratios([Y_train, Y_val, Y_test], ["Y_train", "Y_val", "Y_test"])

    for labelSet in [Y_train, Y_val, Y_test]:
        plt.plot(labelSet)
    #    plt.show()

    print(" ----- NOT NORMALIZED -----")
    tryThisTomorrow(X_train, X_val, X_test)
    print(" ----- NORMALIZED ----- ")
    tryThisTomorrow(X_train_normalized, X_val_normalized, X_test_normalized)
 #   sys.exit()
   # return

    if SIMPLE_layers + GRU_layers + LSTM_layers > 0:
    # Only expand dims for RNNs
        print(f"Before EXPAND DIM --- X_train_normalized.shape = {X_train_normalized.shape}")
        X_train_normalized = np.expand_dims(X_train_normalized, axis=1)
        X_val_normalized = np.expand_dims(X_val_normalized, axis=1)
        X_test_normalized = np.expand_dims(X_test_normalized, axis=1)
        print(f"After EXPAND DIM --- X_train_normalized.shape = {X_train_normalized.shape}")

    # Define the model
    model = tf.keras.Sequential(name='My-NN')
    # Choose model type based on config
    if SIMPLE_layers + GRU_layers + LSTM_layers == 0:
        # 🧠 MLP-only path
        model.add(tf.keras.layers.Input(shape=X_train_normalized.shape[1:], name='input_layer'))
        for i in range(config["layers"]["Dense"]):
            model.add(tf.keras.layers.Dense(dense_neurons[i], activation=activation_dense, kernel_regularizer=kernel_regularizer_dense))
        if config["layers"]["Dropout"] > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    else:
        model.add(tf.keras.layers.Input((X_train_normalized.shape[1],  X_train_normalized.shape[2]), name='input_layer'))

        # Add layers dynamically
        add_rnn_layers(model, "SimpleRNN", SIMPLE_layers, units_simple, GRU_layers > 0 or LSTM_layers > 0)
        add_rnn_layers(model, "GRU", GRU_layers, units_gru, LSTM_layers > 0, recurrent_dropout)
        add_rnn_layers(model, "LSTM", LSTM_layers, units_lstm, False, recurrent_dropout)  # No RNN follows LSTM
        print(f"LSTM Un {units_lstm}")

        if layers["BatchNorm"] > 0:
          model.add(tf.keras.layers.BatchNormalization())

        if layers["Dense"] > 0:
            for i in range(layers["Dense"]): # Iterate over the list instead of passing it directly
                model.add(tf.keras.layers.Dense(dense_neurons[i], activation=activation_dense, kernel_regularizer=kernel_regularizer_dense))

        if layers["Dropout"] > 0:
          model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

    start_time = time.perf_counter() # Get current time at start

  #  learning_rate = 0.007 #0.035 Pitt for ncl=5???
    if optimizer_class == SGD:
        optimizer = optimizer_class(learning_rate=learning_rate, momentum=momentum)
    else:
        optimizer = optimizer_class(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    print(f"\nLearningRate = {learning_rate:.6f}, Optimizer = {optimizer}")

#    X_train_normalized = X_train; Y_train_normalized = Y_train; X_val_normalized = X_val; Y_val_normalized = Y_val
    history = model.fit(X_train_normalized, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_normalized, Y_val), callbacks=[early_stop, lr_scheduler])

    end_time = time.perf_counter() # Get current time at end
    rnn_neural_time = end_time - start_time # Subtract the time at start and time at end, to find the total run time
    print(f"Training Time: {rnn_neural_time:.6f}")

    predictions = model.predict(X_test_normalized)

    # MAYBE DO THIS TO CONTROL PREDICTION
    # Apply default threshold (0.5) if not tuning
    preds_binary = (predictions.flatten() >= 0.5).astype(int)

    np.set_printoptions(formatter={'float': custom_formatter}, linewidth=np.inf)
    print(f'Predictions shape = {predictions.shape}'); print(f'Y_test_normalized shape = {Y_test.shape}')

    print(f'predictions = {predictions.T}'); print(f'real labels = {Y_test}')

    rand_index_pred = 5
    random_numbers = [random.randint(0, Y_test.shape[0]-1) for _ in range(rand_index_pred)]
    for i in random_numbers:
        print(f'\nFor i = {i}, we have:')
        print(f'Y_predictions[i]     = {predictions[i]}')
        print(f'Y_test_normalized[i] = {Y_test[i]}')

    mae = np.mean(np.abs(Y_test - predictions))
    mse = np.mean(np.square(Y_test - predictions))
    loss_evaluate = model.evaluate(X_test_normalized, Y_test)

    # Extract metric names properly
    metric_names = [m.name if hasattr(m, 'name') else m for m in metrics]

    # Format the loss value and the metrics to match their names
    decimalPoints = 6
    formatted_loss = f"loss = {loss_evaluate[0]:.{decimalPoints}f}"  # Format the loss value
    formatted_metrics = {metric: f'{num:.{decimalPoints}f}' for metric, num in zip(metric_names, loss_evaluate[1:])}
    test_metrics = formatted_metrics
    test_metrics['loss'] = f"{loss_evaluate[0]:.{decimalPoints}f}"

    # Now, print each metric with its corresponding value
    formatted_string = ', '.join([f"'{metric}' = {formatted_metrics[metric]}" for metric in formatted_metrics])

    print()
#    print(f'\nManual Calculation -> MAE = {mae:.6f} and MSE = {mse:.6f}')
    print(f'Evaluate number = {formatted_loss}, {formatted_string}\nwhere loss: {loss} and metrics: {metric_names}')
    figureNameParams = f"{needSplitting}split_{scalerName}_ep{epochs}_lr{learning_rate}_batch{batch_size}_activ{activation_dense}"
    print(f"Shape of predictions: {predictions.shape}")
    print(f"Shape of Y_test_normalized: {Y_test.shape}")

    saveTrainingMetricsToFile(history, model, config, learning_rate, optimizer, rnn_neural_time, test_metrics, filepathsAll, predictions.flatten(), Y_test.flatten(), fp.FILEPATH_DATA, figureNameParams, ratio_0_to_1_ALL)
    plotTrainValMetrics(history, fp.FILEPATH_DATA, figureNameParams)


split_options = CONFIG['split_options'] # Define as a variable
for needSplitting in split_options:
    for lr in learning_rate:
        lr = np.round(lr, 8)

        config = CONFIG.copy()
        config["learning_rate"] = lr

        print(f"needSplitting={needSplitting}")
        modelVangRNN = model03_VangRNN(data, labels, needSplitting=needSplitting, config=config)
import sys
import os
import time
import random
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### TensorFlow & Keras Imports ###
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import MeanSquaredError, Accuracy, Precision, Recall
from keras.regularizers import l2

tf.get_logger().setLevel('ERROR')  # Suppress DEBUG logs

### SciKit-Learn Imports ###
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils import resample

from utils00 import (
    returnFilepathToSubfolder,
    doTrainValTestSplit,
    doTrainValTestSplit2,
    plotTrainValMetrics,
    plot_bootstrap_distribution,
    saveTrainingMetricsToFile,
    makeLabelsInt, readCsvAsDataframe
)

import Paths_HELP03 as fp

def print_shapes(name, original, normalized):
    """Helper function to print original and normalized dataset shapes."""
    print(f'{name} shape is = {original.shape}')
    print(f'{name} normalized shape is = {normalized.shape}')

def returnDataLabelsWhenWithoutSignal2Vec(data, labels):
    data = data.dropna().reset_index(drop=True)
    # Ensure labels align with the updated data
    if type(labels) != type(pd.Series):
        labels_s = pd.Series(labels)
      #  labels_s = labels_s.iloc[:, 0]  # convert to series
    labels_s = labels_s[data.index]
    labels_s = labels_s.reset_index(drop=True)

    return data.to_numpy(), labels_s

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

embeddingsPath = "/02_Embeddings/"
folderPath = os.getcwd() + embeddingsPath

#timeSeriesDataPath = "/01_TimeSeriesData/"; embeddingsPath = timeSeriesDataPath; filepath_data = f"Pitt_sR11025.0_2025-01-20_23-11-13_output.csv" #USE THIS TO TEST WITHOUT SIGNAL2VEC
#data = returnData(filepath_data, "allData")

data = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_DATA, "allData")
labels = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_LABELS, "allLabels", as_series=True)
num_classes = len(np.unique(labels))  # Replace with the number of your classes

#data, labels = returnDataLabelsWhenWithoutSignal2Vec(data, labels)

val_ratio = 1

indices_step02_train = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_INDICES_TRAIN, "indicesTrain")
indices_step02_val = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_INDICES_VAL, "indicesVal")
indices_step02_test = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_INDICES_TEST, "indicesTest")
indices_step02 = readCsvAsDataframe(fp.FOLDER_PATH, fp.FILEPATH_INDICES, "indicesAll").to_numpy()

def returnDatasplit(needSplitting = "NO"):
    global val_ratio
    print(f"\n----- NEEDS SPLITTING == {needSplitting} -----")
    if needSplitting == "NO":
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

        X_train, X_val, X_test, Y_train, Y_val, Y_test, val_ratio, indices_train, indices_val, indices_test = doTrainValTestSplit2(X_data, Y_targets)

        caseTypeStrings = ["Train", "Val", "Test"]
        indicesStrings = [indices_train, indices_val, indices_test]
        formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        subfolderName = "03_ClassificationResults"
        for i in range (0, len(caseTypeStrings)):
            df_indices = pd.DataFrame({'Indices': indicesStrings[i]})
            filename = "Indices" + caseTypeStrings[i] + "_" + formatted_datetime + ".csv"
            filenameFull = returnFilepathToSubfolder(filename, subfolderName)
            df_indices.to_csv(filenameFull, index=False, header=False)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, val_ratio, indices_train, indices_val, indices_test


def add_rnn_layers(model, rnn_type, num_layers, units, next_rnn_exists, recurrent_dropout=None):
    rnn_layer = {"SimpleRNN": tf.keras.layers.SimpleRNN, "GRU": tf.keras.layers.GRU, "LSTM": tf.keras.layers.LSTM}.get(rnn_type)
    if not rnn_layer or num_layers <= 0:
        return  # No layers to add

    use_recurrent_dropout = recurrent_dropout if recurrent_dropout is not None and rnn_type in ["GRU", "LSTM"] else 0.0     # Check if recurrent_dropout is specified and valid (only for GRU/LSTM)

    for i in range(num_layers - 1):
        model.add(rnn_layer(units[i] if isinstance(units, list) else units, return_sequences=True, recurrent_dropout=use_recurrent_dropout))

    # The last layer should return sequences only if another RNN follows
    model.add(rnn_layer(units[-1] if isinstance(units, list) else units, return_sequences=next_rnn_exists, recurrent_dropout=use_recurrent_dropout))

def check_indicesEqual(indices_step02, indices_all):
    print(f'Indices shape: step2 = {indices_step02.shape}, step3 = {indices_all.shape}')
    if not np.array_equal(indices_step02, indices_all):
        print(f"Condition failed: {indices_step02[0]} != {indices_all[0]}")
        sys.exit()  # Terminate the execution
    else:
        print(f"Condition success: indices_step02 == indices_all -> {np.array_equal(indices_step02, indices_all)}")

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
    check_indicesEqual(indices_step02, indices_all)

    # save data to a CSV
    subfolderName = "03_ClassificationResults"
    suffix = f"Splitting_{needSplitting}"

    # Save training, validation, and test data using the helper function
    for dataset, label in zip([X_train, X_val, X_test], ["train", "val", "test"]):
        save_data_to_csv(dataset, eval(f"Y_{label}"), subfolderName, suffix, label)

    # Normalize the data
 #   x_scaler = MinMaxScaler()
    x_scaler = StandardScaler()

    X_train_normalized = x_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_normalized = x_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_normalized = x_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    ratio_0_to_1_ALL = calculate_class_ratios([Y_train, Y_val, Y_test], ["Y_train", "Y_val", "Y_test"])

    print(f"X_train_normalized.shape = {X_train_normalized.shape}")
    X_train_normalized = np.expand_dims(X_train_normalized, axis=1)
    X_val_normalized = np.expand_dims(X_val_normalized, axis=1)
    X_test_normalized = np.expand_dims(X_test_normalized, axis=1)
    print(f"X_train_normalized.shape = {X_train_normalized.shape}")

    # Define the model
    model = tf.keras.Sequential(name='my-rnn')
    model.add(tf.keras.layers.Input((X_train_normalized.shape[1],  X_train_normalized.shape[2]), name='input_layer'))

    # Add layers dynamically
    add_rnn_layers(model, "SimpleRNN", SIMPLE_layers, units_simple, GRU_layers > 0 or LSTM_layers > 0)
    add_rnn_layers(model, "GRU", GRU_layers, units_gru, LSTM_layers > 0, recurrent_dropout)
    add_rnn_layers(model, "LSTM", LSTM_layers, units_lstm, False, recurrent_dropout)  # No RNN follows LSTM

    if layers["BatchNorm"] > 0:
      model.add(tf.keras.layers.BatchNormalization())

    if layers["Dense"] > 0:
        for _ in range(0, layers["Dense"]):
          model.add(tf.keras.layers.Dense(dense_neurons, activation=activation_dense, kernel_regularizer=kernel_regularizer_dense))

    if layers["Dropout"] > 0:
      model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

    start_time = time.perf_counter() # Get current time at start

  #  learning_rate = 0.007 #0.035 Pitt for ncl=5???
    momentum = 0.9
 #   optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    optimizer = optimizer_class(learning_rate=learning_rate)
    print(f"LearningRate = {learning_rate}, Optimizer = {optimizer}")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

#    X_train_normalized = X_train; Y_train_normalized = Y_train; X_val_normalized = X_val; Y_val_normalized = Y_val
    # Train the model
    history = model.fit(X_train_normalized, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_normalized, Y_val))

    end_time = time.perf_counter() # Get current time at end
    rnn_neural_time = end_time - start_time # Subtract the time at start and time at end, to find the total run time
    print(f"Training Time: {rnn_neural_time:.6f}")

    predictions = model.predict(X_test_normalized)

    def custom_formatter(x):
        return f"{x:.6f}"
    np.set_printoptions(formatter={'float': custom_formatter}, linewidth=np.inf)
    print(f'Predictions shape = {predictions.shape}'); print(f'Y_test_normalized shape = {Y_test.shape}')

    print(f'predictions = {predictions.T}'); print(f'actual labels = {Y_test}')

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

 #   formatted_loss = [f'{num:.6f}' for num in loss_evaluate]
  #  formatted_string = ', '.join(formatted_loss)
 #   print(f'\nManual Calculation -> MAE = {mae:.6f} and MSE = {mse:.6f}\nEvaluate number = {formatted_string}\nwhere loss: {loss} and metrics: {metrics}')
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
    figureNameParams = f"needSplitting{needSplitting}_ep{epochs}_lr{learning_rate}_batch{batch_size}_activ{activation_dense}"
    print(f"Shape of predictions: {predictions.shape}")
    print(f"Shape of Y_test_normalized: {Y_test.shape}")

    saveTrainingMetricsToFile(history, model, config, learning_rate, optimizer, rnn_neural_time, test_metrics, predictions.flatten(), Y_test.flatten(), fp.FILEPATH_DATA, figureNameParams, ratio_0_to_1_ALL)
    plotTrainValMetrics(history, fp.FILEPATH_DATA, figureNameParams)

### Global Configuration Dictionary ###
CONFIG = {
    "batch_size": 256,
    "epochs": 50,
    "loss": "binary_crossentropy",
    "metrics": ['accuracy', Precision(), Recall()], # metrics = ['mse', 'mae', 'accuracy']
    "optimizer": Adam,
    "units": {
        "SimpleRNN": [32, 32], # [32, 32]
        "GRU": [32], # 32
        "LSTM": [32, 32, 32],  # 32
    },
    "neurons": {
        "Dense": 64 # 64
    },
    "layers": {
        "SimpleRNN": 0, # 2
        "GRU": 1, # 0
        "LSTM": 0, # 1
        "Dense": 1,
        "Dropout": 0,
        "BatchNorm": 0
    },
    "recurrent_dropout": 0.2, # 0.2
    "activation_dense": "relu",
    "dropout": 0.4,
    "kernel_regularizer_dense": l2(0.001) # None
}

lr_min = 0.0005 # 0.001, 0.035
lr_max = 0.095 # 0.01
lr_distinct = 10 # 10
learning_rate = np.linspace(lr_min, lr_max, num=lr_distinct).tolist()
for lr in learning_rate:
    config = CONFIG.copy()
    config["learning_rate"] = lr

    modelVangRNN = model03_VangRNN(data, labels, needSplitting="NO", config=config)
#for lr in learning_rate:
#    modelVangRNN = model03_VangRNN(data, labels, needSplitting="YES", learning_rate = lr)
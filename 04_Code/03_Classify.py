import sys
import os
import time
import random
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### PyTorch Imports ###
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.models import resnet50, ResNet50_Weights, densenet201, DenseNet201_Weights, list_models

### TensorFlow & Keras Imports ###
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.optimizers import SGD
from tensorflow.keras.metrics import MeanSquaredError, Accuracy, Precision, Recall

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

embeddingsPath = "/02_Embeddings/"
folderPath = os.getcwd() + embeddingsPath
filepath_data = "Embeddings_Lu_2025-01-15_23-11-50.csv"
#filepath_data = "Lu_sR50_2025-01-06_01-40-21_output (Copy).csv"
filepath_data = "Embeddings_Pitt_2025-01-21_02-02-38.csv"
filepath_data = "Embeddings_Pitt_2025-01-26_23-29-29.csv"
filepath_data = "Embeddings_Pitt_2025-01-28_00-39-51.csv"
filepath_data = "Embeddings_Pitt_2025-01-29_22-14-49.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-01-30_00-49-18.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-02_16-27-13.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-02_23-37-08.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-22_15-09-08.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-24_00-04-20.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-25_21-15-00.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-02.csv"
filepath_data = "Embeddings_Pitt_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-06.csv"
#filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-20_20-22-02.csv"
#timeSeriesDataPath = "/01_TimeSeriesData/"; embeddingsPath = timeSeriesDataPath; filepath_data = f"Pitt_sR11025.0_2025-01-20_23-11-13_output.csv" #USE THIS TO TEST WITHOUT SIGNAL2VEC
#data = returnData(filepath_data, "allData")
data = readCsvAsDataframe(folderPath, filepath_data, "allData")

embeddingsPath = "/02_Embeddings/"
filepath_labels = "Lu_sR50_2025-01-06_01-40-21_output.csv"
filepath_labels = "Labels_Pitt_2025-01-21_02-05-52.csv"
filepath_labels = "Labels_Pitt_2025-01-26_23-29-29.csv"
filepath_labels = "Labels_Pitt_2025-01-30_00-49-18.csv"
filepath_labels = "Labels_Pitt_2025-02-02_16-27-13.csv"
filepath_labels = "Labels_Pitt_2025-02-02_23-37-08.csv"
filepath_labels = "Labels_Pitt_2025-02-20_20-22-02.csv"
filepath_labels = "Labels_Pitt_2025-02-22_15-09-08.csv"
filepath_labels = "Labels_Pitt_2025-02-24_00-04-20.csv"
filepath_labels = "Labels_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-25_21-15-00.csv"
filepath_labels = "Labels_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-02.csv"
filepath_labels = "Labels_Pitt_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
filepath_labels = "Labels_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv"
filepath_labels = "Labels_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-06.csv"
labels = readCsvAsDataframe(folderPath, filepath_labels, "allLabels", as_series=True)
num_classes = len(np.unique(labels))  # Replace with the number of your classes

#data, labels = returnDataLabelsWhenWithoutSignal2Vec(data, labels)

filepath_data_train = "Embeddings_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv"
filepath_data_train = "Embeddings_Pitt_trainSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
filepath_data_train = "Embeddings_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv"
filepath_data_train = "Embeddings_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-06.csv"
filepath_data_val = "Embeddings_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv"
filepath_data_val = "Embeddings_Pitt_valSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
filepath_data_val = "Embeddings_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv"
filepath_data_val = "Embeddings_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-07.csv"
filepath_data_test = "Embeddings_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv"
filepath_data_test = "Embeddings_Pitt_testSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
filepath_data_test = "Embeddings_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv"
filepath_data_test = "Embeddings_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-07.csv"

filepath_labels_train = "Labels_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv"
filepath_labels_train = "Labels_Pitt_trainSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
filepath_labels_train = "Labels_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv"
filepath_labels_train = "Labels_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-06.csv"
filepath_labels_val = "Labels_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv"
filepath_labels_val = "Labels_Pitt_valSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
filepath_labels_val = "Labels_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv"
filepath_labels_val = "Labels_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-07.csv"
filepath_labels_test = "Labels_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv"
filepath_labels_test = "Labels_Pitt_testSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
filepath_labels_test = "Labels_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv"
filepath_labels_test = "Labels_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-07.csv"

X_train = readCsvAsDataframe(folderPath, filepath_data_train, "X_train").to_numpy()
X_val = readCsvAsDataframe(folderPath, filepath_data_val, "X_val").to_numpy()
X_test = readCsvAsDataframe(folderPath, filepath_data_test, "X_test").to_numpy()
Y_train = readCsvAsDataframe(folderPath, filepath_labels_train, "Y_train", as_series=True)
Y_val = readCsvAsDataframe(folderPath, filepath_labels_val, "Y_val", as_series=True)
Y_test = readCsvAsDataframe(folderPath, filepath_labels_test, "Y_test", as_series=True)
val_ratio = 1

filepath_indices = "Indices_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-06.csv"
filepath_indices_train = "Indices_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-06.csv"
filepath_indices_val = "Indices_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-07.csv"
filepath_indices_test = "Indices_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-07.csv"
indices_step02_train = readCsvAsDataframe(folderPath, filepath_indices_train, "indicesTrain")
indices_step02_val = readCsvAsDataframe(folderPath, filepath_indices_val, "indicesVal")
indices_step02_test = readCsvAsDataframe(folderPath, filepath_indices_test, "indicesTest")
indices_step02 = readCsvAsDataframe(folderPath, filepath_indices, "indicesAll").to_numpy()

def returnDatasplit(needSplitting = "NO"):
    global val_ratio
    print(f"\n----- NEEDS SPLITTING == {needSplitting} -----")
    if needSplitting == "NO":
        X_train = readCsvAsDataframe(folderPath, filepath_data_train, "X_train").to_numpy()
        X_val = readCsvAsDataframe(folderPath, filepath_data_val, "X_val").to_numpy()
        X_test = readCsvAsDataframe(folderPath, filepath_data_test, "X_test").to_numpy()
        Y_train = readCsvAsDataframe(folderPath, filepath_labels_train, "Y_train", as_series=True).to_numpy()
        Y_val = readCsvAsDataframe(folderPath, filepath_labels_val, "Y_val", as_series=True).to_numpy()
        Y_test = readCsvAsDataframe(folderPath, filepath_labels_test, "Y_test", as_series=True).to_numpy()
        indices_train = indices_step02_train.to_numpy(); indices_val = indices_step02_val.to_numpy(); indices_test = indices_step02_test.to_numpy()

    else:
        X_data = np.array(data); Y_targets = np.array(labels)
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


def add_rnn_layers(model, rnn_type, num_layers, units, next_rnn_exists):
    """
    Adds RNN layers to the model.

    Parameters:
        model (tf.keras.Sequential): The Keras model.
        rnn_type (str): 'SimpleRNN', 'GRU', or 'LSTM'.
        num_layers (int): Number of layers to add.
        units (list or int): List of units per layer or single int if fixed.
        next_rnn_exists (bool): Whether another RNN type follows.
    """
    rnn_layer = {
        "SimpleRNN": tf.keras.layers.SimpleRNN,
        "GRU": tf.keras.layers.GRU,
        "LSTM": tf.keras.layers.LSTM
    }.get(rnn_type)

    if not rnn_layer or num_layers <= 0:
        return  # No layers to add

    for i in range(num_layers - 1):
        model.add(rnn_layer(units[i] if isinstance(units, list) else units, return_sequences=True))

    # The last layer should return sequences only if another RNN follows
    model.add(rnn_layer(units[-1] if isinstance(units, list) else units, return_sequences=next_rnn_exists))

def check_indicesEqual(indices_step02, indices_all):
    print(f'Indices shape: step2 = {indices_step02.shape}, step3 = {indices_all.shape}')
    if not np.array_equal(indices_step02, indices_all):
        print(f"Condition failed: {indices_step02} != {indices_all}")
        sys.exit()  # Terminate the execution
    else:
        print(f"Condition success: indices_step02 == indices_all -> {np.array_equal(indices_step02, indices_all)}")

def model03_VangRNN(data, labels, needSplitting, learning_rate = 0.035):
    def f1_score(y_true, y_pred):
        precision = tf.keras.metrics.Precision()
        recall = tf.keras.metrics.Recall()
        precision.update_state(y_true, y_pred)
        recall.update_state(y_true, y_pred)

        # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
        return 2 * (precision.result() * recall.result()) / (precision.result() + recall.result() + tf.keras.backend.epsilon())

    X_train, X_val, X_test, Y_train, Y_val, Y_test, val_ratio, indices_train, indices_val, indices_test = returnDatasplit(needSplitting)
  #  print(X_train[0])
    print(Y_train); print(Y_val); print(Y_test)

    indices_all = np.vstack([indices_train.reshape(-1, 1), indices_val.reshape(-1, 1), indices_test.reshape(-1, 1)])
    check_indicesEqual(indices_step02, indices_all)

    # save data to a CSV
    subfolderName = "03_ClassificationResults"
    suffix = f"Splitting_{needSplitting}"

    # Save training, validation, and test data using the helper function
#    save_data_to_csv(X_train, Y_train, "03_ClassificationResults", suffix, "train")
 #   save_data_to_csv(X_val, Y_val, "03_ClassificationResults", suffix, "val")
#    save_data_to_csv(X_test, Y_test, "03_ClassificationResults", suffix, "test")
    for dataset, label in zip([X_train, X_val, X_test], ["train", "val", "test"]):
        save_data_to_csv(dataset, eval(f"Y_{label}"), subfolderName, suffix, label)

    # Normalize the data
  #  x_scaler = MinMaxScaler()
    x_scaler = StandardScaler()

    X_train_normalized = x_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_normalized = x_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_normalized = x_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

 #   print(type(X_test),type(X_test_normalized))
    # Count occurrences
    allYs = [Y_train, Y_val, Y_test]
    allYNames = ["Y_train", "Y_val", "Y_test"]
    ratio_0_to_1_ALL= []
    for Y in range(len(allYs)):
        count_0 = np.sum(allYs[Y] == 0)
        count_1 = np.sum(allYs[Y] == 1)

        # Ratio (0s to 1s)
        ratio_0_to_1 = count_0 / count_1
        print(f"{allYNames[Y]}, Y_0/Y_1 = {ratio_0_to_1}")
        ratio_0_to_1_ALL.append(ratio_0_to_1)

    # Normalize Labels (might not be necessary)
    y_scaler = MinMaxScaler()
    Y_train_normalized = y_scaler.fit_transform(Y_train.reshape(-1, 1)) if Y_train.ndim == 1 else y_scaler.fit_transform(Y_train)
    Y_val_normalized = y_scaler.transform(Y_val.reshape(-1, 1)) if Y_train.ndim == 1 else y_scaler.transform(Y_val)
    Y_test_normalized = y_scaler.transform(Y_test.reshape(-1, 1)) if Y_train.ndim == 1 else y_scaler.transform(Y_test)

    # Print shapes for training and test sets
 #   print_shapes("X_train", X_train, X_train_normalized)
 #   print_shapes("X_test", X_test, X_test_normalized)
 #   print_shapes("Y_train", Y_train, Y_train_normalized)
 #   print_shapes("Y_test", Y_test, Y_test_normalized)
    # Print validation shapes only if val_ratio > 0
#    if val_ratio > 0:
#        print_shapes("X_val", X_val, X_val_normalized)
#        print_shapes("Y_val", Y_val, Y_val_normalized)

    loss = 'binary_crossentropy'#'mae'
    metrics = ['mse', 'mae', 'accuracy']
    metrics = ['accuracy', Precision(), Recall()]
    batch_size = 32
    epochs = 50 # 50

    units_simple = [32, 32] # [32, 32]
    units_lstm = [32, 32, 32] # 32
    units_gru = 32
    neurons_dense = 64

  #  dropout = 0.4
    activation_dense = 'sigmoid'

    LSTM_type = 'YES'
    GRU_type = 'YES'
    SIMPLE_type = 'YES'
    SIMPLE_layers = 2 # 2
    GRU_layers = 0 # 0
    LSTM_layers = 1 # 1
    Dropout_layers = 1
    BatchNorm_layers = 1

    ExtraDense = 'YES'
    DENSE_layers = 1

    print(X_train_normalized.shape)
    X_train_normalized = np.expand_dims(X_train_normalized, axis=1)
    X_val_normalized = np.expand_dims(X_val_normalized, axis=1)
    X_test_normalized = np.expand_dims(X_test_normalized, axis=1)
    print(X_train_normalized.shape)


    # Define the model
    model = tf.keras.Sequential(name='my-rnn')

    #inputs = tf.keras.Input(shape=(X_train_normalized.shape[1], X_train_normalized.shape[2]))
    #model.add(inputs)  # Add the input layer

    model.add(tf.keras.layers.Input((X_train_normalized.shape[1],  X_train_normalized.shape[2]), name='input_layer'))

    # Add layers dynamically
    add_rnn_layers(model, "SimpleRNN", SIMPLE_layers, units_simple, GRU_type == "YES" or LSTM_type == "YES")
    add_rnn_layers(model, "GRU", GRU_layers, units_gru, LSTM_type == "YES")
    add_rnn_layers(model, "LSTM", LSTM_layers, units_lstm, False)  # No RNN follows LSTM

  #  if Dropout_layers == 1:
 #     model.add(tf.keras.layers.Dropout(dropout))

 #   if BatchNorm_layers == 1:
  #    model.add(tf.keras.layers.BatchNormalization())

    if ExtraDense == 'YES':
        for i in range(0, DENSE_layers):
          model.add(tf.keras.layers.Dense(neurons_dense, activation=activation_dense))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

    start_time = time.perf_counter() # Get current time at start

    # Compile the model
  #  learning_rate = 0.007 #0.035 Pitt for ncl=5???
    momentum = 0.9
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
  #  optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    # Summary of the model
    model.summary()

#    X_train_normalized = X_train; Y_train_normalized = Y_train; X_val_normalized = X_val; Y_val_normalized = Y_val
    print("NO NEED TO SCALE Y, SO OVERRIDING THE VALUES")
 #   print(Y_train, Y_val, Y_test)
    Y_train_normalized = Y_train; Y_val_normalized = Y_val; Y_test_normalized = Y_test
  #  print(Y_train, Y_val, Y_test)
    # Train the model
    history = model.fit(X_train_normalized, Y_train_normalized, epochs=epochs, batch_size=batch_size, validation_data=(X_val_normalized, Y_val_normalized))
    # batch size of 1 (i.e., updating the model's weights after every single sample)

    end_time = time.perf_counter() # Get current time at end
    rnn_neural_time = end_time - start_time # Subtract the time at start and time at end, to find the total run time
    print(f"Training Time: {rnn_neural_time:.6f}")

    predictions = model.predict(X_test_normalized)

    def custom_formatter(x):
        return f"{x:.6f}"
    np.set_printoptions(formatter={'float': custom_formatter}, linewidth=np.inf)
    print(f'Predictions shape = {predictions.shape}')
    print(f'Y_test_normalized shape = {Y_test_normalized.shape}')

    print(f'predictions = {predictions.T}')
    print(f'actual labels = {Y_test_normalized}')

    rand_index_pred = 5
    random_numbers = [random.randint(0, Y_test_normalized.shape[0]-1) for _ in range(5)]
    for i in random_numbers:
        print(f'\nFor i = {i}, we have:')
        print(f'Y_predictions[i]     = {predictions[i]}')
        print(f'Y_test_normalized[i] = {Y_test_normalized[i]}')

    mae = np.mean(np.abs(Y_test_normalized - predictions))
    mse = np.mean(np.square(Y_test_normalized - predictions))
    loss_evaluate = model.evaluate(X_test_normalized, Y_test_normalized)

 #   formatted_loss = [f'{num:.6f}' for num in loss_evaluate]
  #  formatted_string = ', '.join(formatted_loss)
 #   print(f'\nManual Calculation -> MAE = {mae:.6f} and MSE = {mse:.6f}\nEvaluate number = {formatted_string}\nwhere loss: {loss} and metrics: {metrics}')
    # Format the loss value and the metrics to match their names
    decimalPoints = 6
    formatted_loss = f"loss = {loss_evaluate[0]:.{decimalPoints}f}"  # Format the loss value
    formatted_metrics = {metric: f'{num:.{decimalPoints}f}' for metric, num in zip(metrics, loss_evaluate[1:])}
    test_metrics = formatted_metrics
    test_metrics['loss'] = f"{loss_evaluate[0]:.{decimalPoints}f}"

    # Now, print each metric with its corresponding value
    formatted_string = ', '.join([f"'{metric}' = {formatted_metrics[metric]}" for metric in formatted_metrics])

    print()
#    print(f'\nManual Calculation -> MAE = {mae:.6f} and MSE = {mse:.6f}')
    print(f'Evaluate number = {formatted_loss}, {formatted_string}\nwhere loss: {loss} and metrics: {metrics}')
    figureNameParams = f"needSplitting{needSplitting}_ep{epochs}_lr{learning_rate}_batch{batch_size}_activ{activation_dense}"
    print(f"Shape of predictions: {predictions.shape}")
    print(f"Shape of Y_test_normalized: {Y_test_normalized.shape}")

    saveTrainingMetricsToFile(history, model, rnn_neural_time, test_metrics, predictions.flatten(), Y_test_normalized.flatten(), filepath_data, figureNameParams, ratio_0_to_1_ALL)
    plotTrainValMetrics(history, filepath_data, figureNameParams)

lr_min = 0.035 # 0.001
lr_max = 0.1 # 0.01
lr_distinct = 1 # 10
learning_rate = np.linspace(lr_min, lr_max, num=lr_distinct).tolist()
for lr in learning_rate:
    modelVangRNN = model03_VangRNN(data, labels, needSplitting="NO" , learning_rate = lr)
for lr in learning_rate:
    modelVangRNN = model03_VangRNN(data, labels, needSplitting="YES", learning_rate = lr)
import time
from datetime import datetime
import random
import torch
import torch.optim as optim
import torchvision
from keras.optimizers import SGD
from pyparsing import pyparsing_test
from torchvision.models import resnet50, ResNet50_Weights, list_models, densenet201, DenseNet201_Weights
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanSquaredError, Accuracy, Precision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
tf.get_logger().setLevel('ERROR')  # Suppress DEBUG logs
from utils00 import returnFilepathToSubfolder, doTrainValTestSplit, plotTrainValMetrics, plot_bootstrap_distribution


def compute_confidence_interval(model, X_test, Y_test, n_bootstrap=1000, ci=95, random_state = 0):
    """
    Compute confidence interval for a metric (e.g., accuracy) using bootstrap sampling.

    Parameters:
    - model: Trained model.
    - X_test: Test features.
    - Y_test: True labels for the test data.
    - n_bootstrap: Number of bootstrap samples.
    - ci: Desired confidence interval (e.g., 95).

    Returns:
    - lower_bound: Lower bound of the confidence interval.
    - upper_bound: Upper bound of the confidence interval.
    """
    bootstrap_accuracies = []

    # Perform bootstrap sampling
    for i in range(n_bootstrap):
        # Check every 10th iteration
        if (i + 1) % 50 == 0:
            print(f"Bootstrap iteration {i + 1} completed")

        # Resample the data with replacement
        X_resampled, Y_resampled = resample(X_test, Y_test, random_state=random_state)

        # Predict on the resampled data
        predictions = model.predict(X_resampled, verbose=0)

        predictions = (predictions > 0.5).astype(int)  # Convert to 0 or 1 based on threshold 0.5

        # Compute the accuracy
        accuracy = accuracy_score(Y_resampled, predictions)

        # Store the accuracy
        bootstrap_accuracies.append(accuracy)

    # Compute the lower and upper percentiles for the confidence interval
    lower_bound = np.percentile(bootstrap_accuracies, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_accuracies, 100 - (100 - ci) / 2)

    return lower_bound, upper_bound, bootstrap_accuracies

def returnData(filepath_data):
    totalpath_data = abspath + embeddingsPath + filepath_data
    data = pd.read_csv(totalpath_data, header=None)
    print(data.shape)
    return data

def returnLabels(filepath_label):
    initial_labels = returnData(filepath_label)

    if type(initial_labels) != type(pd.Series):
#        print(f"type is: {type(initial_labels)}")
        initial_labels = initial_labels.iloc[:, 0]  # convert to series

    labels = initial_labels.to_numpy()
    print(labels)
    return labels


def returnTrainValTest(path_train, path_val, path_test, path_train_labels, path_val_labels, path_test_labels):
    totalpath_train = abspath + embeddingsPath + path_train
    totalpath_val = abspath + embeddingsPath + path_val
    totalpath_test = abspath + embeddingsPath + path_test
    X_train = pd.read_csv(totalpath_train, header=None)
    X_val = pd.read_csv(totalpath_val, header=None)
    X_test = pd.read_csv(totalpath_test, header=None)

    return X_train, X_val, X_test


abspath = "/home/vang/Downloads/"
abspath = os.getcwd()
embeddingsPath = "/02_Embeddings/"
filepath_data = "Embeddings_Lu_2025-01-15_23-11-50.csv"
#filepath_data = "Lu_sR50_2025-01-06_01-40-21_output (Copy).csv"
filepath_data = "Embeddings_Pitt_2025-01-21_02-02-38.csv"
filepath_data = "Embeddings_Pitt_2025-01-26_23-29-29.csv"
filepath_data = "Embeddings_Pitt_2025-01-28_00-39-51.csv"
filepath_data = "Embeddings_Pitt_2025-01-29_22-14-49.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-01-30_00-49-18.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-02_16-27-13.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-02_23-37-08.csv"
# filepath_data = "Pitt_sR11025.0_2025-01-20_23-11-13_output.csv" --- USE THIS TO TEST WITHOUT SIGNAL2VEC
data = returnData(filepath_data)

# Drop NaN rows from data, # Reset indices after dropping rows
# data = data.dropna().reset_index(drop=True)

filepath_labels = "Lu_sR50_2025-01-06_01-40-21_output.csv"
filepath_labels = "Labels_Pitt_2025-01-21_02-05-52.csv"
filepath_labels = "Labels_Pitt_2025-01-26_23-29-29.csv"
filepath_labels = "Labels_Pitt_2025-01-30_00-49-18.csv"
filepath_labels = "Labels_Pitt_2025-02-02_16-27-13.csv"
filepath_labels = "Labels_Pitt_2025-02-02_23-37-08.csv"
labels = returnLabels(filepath_labels)
num_classes = len(np.unique(labels))  # Replace with the number of your classes

#all_models = list_models()
#classification_models = list_models(module=torchvision.models)
#print(classification_models)

filepath_train = "Embeddings_Pitt_train_nCl8_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-02_23-13-30.csv"
filepath_val = "Embeddings_Pitt_val_nCl8_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-02_23-13-30.csv"
filepath_test = "Embeddings_Pitt_test_nCl8_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-02_23-13-30.csv"
filepath_labels_train = "Labels_Pitt_train_2025-02-02_23-13-30.csv"
filepath_labels_val = "Labels_Pitt_val_2025-02-02_23-13-30.csv"
filepath_labels_test = "Labels_Pitt_test_2025-02-02_23-13-30.csv"
X_train = returnData(filepath_train).to_numpy()
X_val = returnData(filepath_val).to_numpy()
X_test = returnData(filepath_test).to_numpy()
Y_train = returnLabels(filepath_labels_train)
Y_val = returnLabels(filepath_labels_val)
Y_test = returnLabels(filepath_labels_test)
val_ratio = 1

def model01_Resnet50(data, labels):
    # Assume `features` is a torch.Tensor with the same dimensions as expected by the ResNet model
    # Example: `features` should have the shape (N, C, H, W), where
    # - N is the batch size,
    # - C is the number of channels (3 for RGB images),
    # - H is the height,
    # - W is the width.

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Ensure your features are on the same device as the model (e.g., CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #features = torch.tensor(data.values, dtype=torch.float32)  # Convert DataFrame to tensor
    #features = features.to(device)

    # Convert to tensor and reshape to pseudo-images
    features = torch.tensor(data.values, dtype=torch.float32)  # (54, 100)
    features = features.view(-1, 1, 10, 10)  # Reshape to (N, C, H, W)

    # Modify the first convolutional layer to accept 1 channel instead of 3
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Modify the fully connected layer for your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.long)  # Convert to PyTorch tensor

    # Example training loop
    model.train()
    for epoch in range(3):  # Replace with the number of epochs you want
        start_time = time.time()  # Start timer
        optimizer.zero_grad()
        outputs = model(features)  # Assuming `features` is your training data
        loss = criterion(outputs, labels)  # Assuming `labels` are your true labels
        loss.backward()
        optimizer.step()
        end_time = time.time()  # End timer

        epoch_time = end_time - start_time  # Calculate epoch time
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, Time: {epoch_time:.2f} seconds")


    # Assuming you have a list of your class names
    custom_classes = ["Control", "Dementia"]  # Replace with your actual class names
    assert len(custom_classes) == num_classes, "Mismatch between custom_classes and num_classes!"


    # Perform predictions
    model.eval()
    with torch.no_grad():
        predictions = model(features).softmax(dim=1)
        for idx, prediction in enumerate(predictions):
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            print(f"Sample {idx}: {custom_classes[class_id]}: {100 * score:.1f}%")

    return model


    # Step 3: Use the model and print the predicted category
    # Ensure the features have the shape (N, 3, H, W) as required by the model
    #prediction = model(features).softmax(dim=1)  # dim=1 as predictions are per class
    #class_ids = prediction.argmax(dim=1)  # Get the class IDs for each item in the batch

    #for idx, class_id in enumerate(class_ids):
    #    score = prediction[idx, class_id].item()
    #    category_name = weights.meta["categories"][class_id]
    #    print(f"Sample {idx}: {category_name}: {100 * score:.1f}%")

def model02_Densenet201(data, labels):
    # Step 1: Initialize model with the best available weights
    weights = DenseNet201_Weights.DEFAULT
    model = densenet201(weights=weights)
    model.eval()

    # Step 2: Ensure your features are on the same device as the model (e.g., CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Convert to tensor and reshape to pseudo-images
    features = torch.tensor(data.values, dtype=torch.float32)  # (54, 100)
    features = features.view(-1, 1, 10, 10)  # Reshape to (N, C, H, W)

    # Resize input to 224x224
    features = F.interpolate(features, size=(224, 224), mode='bilinear', align_corners=False)
    features = features.to(device)

    # Modify the first convolutional layer to accept 1 channel instead of 3
    model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify the classifier layer for your dataset
    num_classes = 2  # Set this according to your dataset
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.long)  # Convert to PyTorch tensor
    labels = labels.to(device)

    # Example training loop
    model.train()
    for epoch in range(3):  # Replace with the number of epochs you want
        start_time = time.time()  # Start timer
        optimizer.zero_grad()
        outputs = model(features)  # Assuming `features` is your training data
        loss = criterion(outputs, labels)  # Assuming `labels` are your true labels
        loss.backward()
        optimizer.step()
        end_time = time.time()  # End timer

        epoch_time = end_time - start_time  # Calculate epoch time
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, Time: {epoch_time:.2f} seconds")

    # Assuming you have a list of your class names
    custom_classes = ["Control", "Dementia"]  # Replace with your actual class names
    assert len(custom_classes) == num_classes, "Mismatch between custom_classes and num_classes!"

    # Perform predictions
    model.eval()
    with torch.no_grad():
        predictions = model(features).softmax(dim=1)
        for idx, prediction in enumerate(predictions):
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            print(f"Sample {idx}: {custom_classes[class_id]}: {100 * score:.1f}%")

    return model

def model03_VangRNN(data, labels):
    def f1_score(y_true, y_pred):
        precision = tf.keras.metrics.Precision()
        recall = tf.keras.metrics.Recall()

        precision.update_state(y_true, y_pred)
        recall.update_state(y_true, y_pred)

        precision_value = precision.result()
        recall_value = recall.result()

        # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision_value * recall_value) / (precision_value + recall_value + tf.keras.backend.epsilon())
        return f1

    X_data = np.array(data); Y_targets = np.array(labels)
    print(f'\nLength of X is = {len(X_data)}. Length of Y is = {len(Y_targets)}')

    X_train, X_val, X_test, Y_train, Y_val, Y_test, val_ratio = doTrainValTestSplit(X_data, Y_targets)

    # Normalize the data
    x_scaler = MinMaxScaler()
    x_scaler = StandardScaler()

#    X_train_normalized = np.array(x_scaler.fit_transform(X_train.shape(-1,1)))
 #   X_val_normalized = x_scaler.transform
#    X_test_normalized = x_scaler.transform
    X_train_normalized = x_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_normalized = x_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_normalized = x_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    print(type(X_test),type(X_test_normalized))

    # Assuming y_train, y_val, and y_test need to be scaled
    y_scaler = MinMaxScaler()

    # Logic for predicting both 1 feature and many features
    if Y_train.ndim == 1:
        Y_train_normalized = y_scaler.fit_transform(Y_train.reshape(-1, 1))
        Y_val_normalized = y_scaler.transform(Y_val.reshape(-1, 1))
        Y_test_normalized = y_scaler.transform(Y_test.reshape(-1, 1))
    else:
        Y_train_normalized = y_scaler.fit_transform(Y_train)  # .reshape(-1, 1))
        Y_val_normalized = y_scaler.transform(Y_val)  # .reshape(-1, 1))
        Y_test_normalized = y_scaler.transform(Y_test)  # .reshape(-1, 1))

    print(f'X_train shape is = {X_train.shape}')
    print(f'X_train normalized shape is = {X_train_normalized.shape}')
    print(f'X_test shape is = {X_test.shape}')
    print(f'X_test normalized shape is = {X_test_normalized.shape}')
    if val_ratio > 0:
        print(f'X_val shape is = {X_val.shape}')
        print(f'X_val normalized shape is = {X_val_normalized.shape}')
    # print(Y_test.shape)

    print(f'\nY_train shape is = {Y_train.shape}')
    print(f'Y_train normalized shape is = {Y_train_normalized.shape}')
    print(f'Y_test shape is = {Y_test.shape}')
    print(f'Y_test normalized shape is = {Y_test_normalized.shape}')
    if val_ratio > 0:
        print(f'Y_val shape is = {Y_val.shape}')
        print(f'Y_val normalized shape is = {Y_val_normalized.shape}')

    X_train_initial = X_train
    X_test_initial = X_test
    Y_train_initial = Y_train
    Y_test_initial = Y_test

    loss = 'binary_crossentropy'#'mae'
    metrics = ['mse', 'mae', 'accuracy']
    batch_size = 32
    epochs = 50

    units_simple = [32, 32]
    units_lstm = 32
    units_gru = 32
    #units_simple = units_lstm = units_gru
    neurons_dense = 32

    dropout = 0.4
    activation_dense = 'sigmoid'

    LSTM_type = 'YES'
    GRU_type = 'YES'
    SIMPLE_type = 'YES'
    SIMPLE_layers = 2
    GRU_layers = 1
    LSTM_layers = 1
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

    if SIMPLE_type == 'YES':
      for i in range(0, SIMPLE_layers - 1):
        model.add(tf.keras.layers.SimpleRNN(units_simple[i], return_sequences=True))
      if GRU_type == 'YES' or LSTM_type == 'YES':
        model.add(tf.keras.layers.SimpleRNN(units_simple[-1], return_sequences=True))
      else:
        model.add(tf.keras.layers.SimpleRNN(units_simple, return_sequences=False))

    if GRU_type == 'YES':
      for i in range(0, GRU_layers - 1):
        model.add(tf.keras.layers.GRU(units_gru, return_sequences=True))
      if LSTM_type == 'YES':
        model.add(tf.keras.layers.GRU(units_gru, return_sequences=True))
      else:
        model.add(tf.keras.layers.GRU(units_gru, return_sequences=False))

    if LSTM_type == 'YES':
      for i in range(0, LSTM_layers - 1):
        model.add(tf.keras.layers.LSTM(units_lstm, return_sequences=True))
      model.add(tf.keras.layers.LSTM(units_lstm, return_sequences=False))
      # return_sequences=True necessary to pass information to next LSTM layer. return_sequences=False typically for final LSTM layer



    #if Dropout_layers == 1:
    #  model.add(tf.keras.layers.Dropout(dropout))

    #if BatchNorm_layers == 1:
    #  model.add(tf.keras.layers.BatchNormalization())

    if ExtraDense == 'YES':
        for i in range(0, DENSE_layers):
          model.add(tf.keras.layers.Dense(neurons_dense, activation=activation_dense))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

    start_time = time.perf_counter() # Get current time at start

    # Compile the model
    learning_rate = 0.05
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
    Y_train_normalized = Y_train; Y_val_normalized = Y_val; Y_test_normalized = Y_test;
 #   print(Y_train, Y_val, Y_test)
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
    formatted_loss = f"loss = {loss_evaluate[0]:.6f}"  # Format the loss value
    formatted_metrics = {metric: f'{num:.6f}' for metric, num in zip(metrics, loss_evaluate[1:])}

    # Now, print each metric with its corresponding value
    formatted_string = ', '.join([f"'{metric}' = {formatted_metrics[metric]}" for metric in formatted_metrics])

    print()
#    print(f'\nManual Calculation -> MAE = {mae:.6f} and MSE = {mse:.6f}')
    print(f'Evaluate number = {formatted_loss}, {formatted_string}\nwhere loss: {loss} and metrics: {metrics}')

    plotTrainValMetrics(history, filepath_data)

    # Example usage:
    n_bootstrap = 1000
    ci = 95
  #  lower_bound, upper_bound, bootstrap_accuracies = compute_confidence_interval(model, X_test_normalized, Y_test_normalized, n_bootstrap, ci)
  #  print(f"Bootstrap Accuracy: {np.mean(bootstrap_accuracies) * 100:.1f}% ± {upper_bound - lower_bound:.4f}%")
   # plot_bootstrap_distribution(bootstrap_accuracies, lower_bound, upper_bound) # Plot the distribution

#modelResnet50 = model01_Resnet50(data, labels)
#modelDensenet01 = model02_Densenet201(data, labels)
modelVangRNN = model03_VangRNN(data, labels)
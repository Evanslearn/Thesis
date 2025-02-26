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
from tensorflow.keras.metrics import MeanSquaredError, Accuracy, Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
tf.get_logger().setLevel('ERROR')  # Suppress DEBUG logs
from utils00 import returnFilepathToSubfolder, doTrainValTestSplit, plotTrainValMetrics, plot_bootstrap_distribution, \
    saveTrainingMetricsToFile, makeLabelsInt, doTrainValTestSplit222222


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
    print(f"data shape = {data.shape}")
    return data

def returnLabels(filepath_label):
    initial_labels = returnData(filepath_label)

    if type(initial_labels) != type(pd.Series):
#        print(f"type is: {type(initial_labels)}")
        initial_labels = initial_labels.iloc[:, 0]  # convert to series

    labels = initial_labels.to_numpy()
  #  print(labels)
    return labels

def returnDataLabelsWhenWithoutSignal2Vec(data, labels):
    data = data.dropna().reset_index(drop=True)
    # Ensure labels align with the updated data
    if type(labels) != type(pd.Series):
        labels_s = pd.Series(labels)
      #  labels_s = labels_s.iloc[:, 0]  # convert to series
    labels_s = labels_s[data.index]
    labels_s = labels_s.reset_index(drop=True)

    return data.to_numpy(), labels_s

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
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-22_15-09-08.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-24_00-04-20.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-25_21-15-00.csv"
filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-02.csv"
filepath_data = "Embeddings_Pitt_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
#filepath_data = "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-20_20-22-02.csv"
#timeSeriesDataPath = "/01_TimeSeriesData/"; embeddingsPath = timeSeriesDataPath; filepath_data = f"Pitt_sR11025.0_2025-01-20_23-11-13_output.csv" #USE THIS TO TEST WITHOUT SIGNAL2VEC
data = returnData(filepath_data)

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
labels = returnLabels(filepath_labels)
num_classes = len(np.unique(labels))  # Replace with the number of your classes

#data, labels = returnDataLabelsWhenWithoutSignal2Vec(data, labels)

filepath_data_train = "Embeddings_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv"
filepath_data_train = "Embeddings_Pitt_trainSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
filepath_data_val = "Embeddings_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv"
filepath_data_val = "Embeddings_Pitt_valSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
filepath_data_test = "Embeddings_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv"
filepath_data_test = "Embeddings_Pitt_testSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"

filepath_labels_train = "Labels_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv"
filepath_labels_train = "Labels_Pitt_trainSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
filepath_labels_val = "Labels_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv"
filepath_labels_val = "Labels_Pitt_valSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
filepath_labels_test = "Labels_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv"
filepath_labels_test = "Labels_Pitt_testSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv"
X_train = returnData(filepath_data_train).to_numpy()
X_val = returnData(filepath_data_val).to_numpy()
X_test = returnData(filepath_data_test).to_numpy()
Y_train = returnLabels(filepath_labels_train)
Y_val = returnLabels(filepath_labels_val)
Y_test = returnLabels(filepath_labels_test)
val_ratio = 1

def returnDatasplit(needSplitting = "NO"):
    global val_ratio
    if needSplitting == "NO":
        X_train = returnData(filepath_data_train).to_numpy()
        X_val = returnData(filepath_data_val).to_numpy()
        X_test = returnData(filepath_data_test).to_numpy()
        Y_train = returnLabels(filepath_labels_train)
        Y_val = returnLabels(filepath_labels_val)
        Y_test = returnLabels(filepath_labels_test)
    else:
        X_data = np.array(data); Y_targets = np.array(labels)
        print(f'\nLength of X is = {len(X_data)}. Length of Y is = {len(Y_targets)}')

        X_train, X_val, X_test, Y_train, Y_val, Y_test, val_ratio = doTrainValTestSplit222222(X_data, Y_targets)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, val_ratio

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

    # Example usage:
  #  n_bootstrap = 1000
 #   ci = 95
  #  lower_bound, upper_bound, bootstrap_accuracies = compute_confidence_interval(model, X_test_normalized, Y_test_normalized, n_bootstrap, ci)
  #  print(f"Bootstrap Accuracy: {np.mean(bootstrap_accuracies) * 100:.1f}% ± {upper_bound - lower_bound:.4f}%")
   # plot_bootstrap_distribution(bootstrap_accuracies, lower_bound, upper_bound) # Plot the distribution

#modelResnet50 = model01_Resnet50(data, labels)
#modelDensenet01 = model02_Densenet201(data, labels)
lr_min = 0.001 # 0.001
lr_max = 0.01 # 0.01
lr_distinct = 1 # 10
learning_rate = np.linspace(lr_min, lr_max, num=lr_distinct).tolist()
for lr in learning_rate:
    print("Loop1")
   # modelVangRNN = model03_VangRNN(data, labels, needSplitting="NO" , learning_rate = lr)
for lr in learning_rate:
    print("Loop2")
   # modelVangRNN = model03_VangRNN(data, labels, needSplitting="YES", learning_rate = lr)
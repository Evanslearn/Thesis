import os
from datetime import datetime
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

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

def doTrainValTestSplit(X_data, Y_targets, test_val_ratio = 0.3, valRatio_fromTestVal = 0.5, random_state = 0):
    # split into train + (val_test) sets
    X_train, X_test_val, Y_train, Y_test_val = train_test_split(X_data, Y_targets, test_size=test_val_ratio,
                                                                random_state=random_state)#, stratify=Y_targets)

    val_ratio = test_val_ratio * valRatio_fromTestVal
    print(
        f'''We have used {test_val_ratio * 100}% of the data for the test+val set. So now, the val_ratio = {valRatio_fromTestVal * 100}%
    of the val-test data, translates to {val_ratio * 100}% of the total data.''')
    # split into val + test sets
    X_test, X_val, Y_test, Y_val = train_test_split(X_test_val, Y_test_val, test_size=valRatio_fromTestVal,
                                                      random_state=random_state)#, stratify=Y_test_val)

    print(f'train - {Y_train}, \nval - {Y_val},\ntest {Y_test}')

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, val_ratio

    # 2. Split into train and (test + val) sets, while stratifying by Y_targets
#    train_indices, test_val_indices = train_test_split(indices, test_size=test_val_ratio, random_state=random_state, stratify=Y_targets)

    # 3. Further split the test_val_indices into validation and test sets
#    val_indices, test_indices = train_test_split(test_val_indices, test_size=valRatio_fromTestVal, random_state=random_state, stratify=Y_targets[test_val_indices])

    # 4. Use the saved indices to select the corresponding data for each split
  #  X_train = X_data[train_indices]
   # Y_train = Y_targets[train_indices]
  #  X_val = X_data[val_indices]
  #  Y_val = Y_targets[val_indices]
 #   X_test = X_data[test_indices]
 #   Y_test = Y_targets[test_indices]


def plot_bootstrap_distribution(bootstrap_accuracies, lower_bound, upper_bound):
    plt.hist(bootstrap_accuracies, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=2, label=f'Lower bound: {lower_bound:.2f}')
    plt.axvline(upper_bound, color='green', linestyle='dashed', linewidth=2, label=f'Upper bound: {upper_bound:.2f}')
    plt.axvline(np.mean(bootstrap_accuracies), color='orange', linestyle='dashed', linewidth=2,
                label=f'Mean: {np.mean(bootstrap_accuracies):.2f}')
    plt.legend()
    plt.title('Bootstrap Distribution of Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.show()


def plotTrainValMetrics(history, filepath_data, figureNameParams):
    # Access metrics from the history
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    training_mae = history.history['mae']
    validation_mae = history.history['val_mae']
    training_mse = history.history['mse']
    validation_mse = history.history['val_mse']

    # Get the number of epochs from the length of the accuracy history
    epochs = len(training_accuracy)

    # Create a 2x2 grid for subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot training and validation accuracy
    axes[0, 0].plot(range(1, epochs + 1), training_accuracy, label='Training Accuracy', color='blue', marker='o')
    axes[0, 0].plot(range(1, epochs + 1), validation_accuracy, label='Validation Accuracy', color='orange', marker='o')
    axes[0, 0].set_title('Accuracy vs Epoch')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot training and validation loss
    axes[0, 1].plot(range(1, epochs + 1), training_loss, label='Training Loss', color='blue', marker='o')
    axes[0, 1].plot(range(1, epochs + 1), validation_loss, label='Validation Loss', color='orange', marker='o')
    axes[0, 1].set_title('Loss vs Epoch')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot training and validation MAE
    axes[1, 0].plot(range(1, epochs + 1), training_mae, label='Training MAE', color='blue', marker='o')
    axes[1, 0].plot(range(1, epochs + 1), validation_mae, label='Validation MAE', color='orange', marker='o')
    axes[1, 0].set_title('MAE vs Epoch')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot training and validation MSE
    axes[1, 1].plot(range(1, epochs + 1), training_mse, label='Training MSE', color='blue', marker='o')
    axes[1, 1].plot(range(1, epochs + 1), validation_mse, label='Validation MSE', color='orange', marker='o')
    axes[1, 1].set_title('MSE vs Epoch')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    filenameFull = returnFileNameToSave(filepath_data, figureNameParams)

    plt.savefig(filenameFull)  # Save the plot using the dynamic filename
    print(filenameFull)

    plt.show()

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
    else:
        fileExtension = "csv"
    # Define the new filename for saving the plot
    save_filename = f"figure_{new_dynamic_filename}.{fileExtension}"  # Save as PNG

    subfolderName = "03_ClassificationResults"
    filenameFull = returnFilepathToSubfolder(save_filename, subfolderName)
    return filenameFull

def saveTrainingMetricsToFile(history, model, training_time, test_metrics, predictions, actual_labels, filepath_data, fileNameParams):
    filenameFull = returnFileNameToSave(filepath_data, fileNameParams, imageflag="NO")

    # Convert history.history (dictionary) to DataFrame
    df_history = pd.DataFrame(history.history)
    df_history.insert(0, "Epoch", range(1, len(df_history) + 1)) # Add epoch numbers

    # Convert test results to DataFrame
    df_results = pd.DataFrame({
        "Prediction": predictions,
        "Actual Label": actual_labels
    })

    # Save everything into a single CSV file
    with open(filenameFull, "w") as f:
        f.write("Training History:\n")
        df_history.to_csv(f, index=False)

        f.write("\nModel Architecture:\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))

        f.write("\nTraining Time:\n")
        f.write(f"{training_time:.6f} seconds\n")

        f.write("\nTest Set Metrics:\n")
        for metric_name, metric_value in test_metrics.items():
            f.write(f"{metric_name},{metric_value}\n")

        f.write("\nPredictions vs Actual Labels:\n")
        df_results.to_csv(f, index=False)

        f.write("\nRandom Sample Comparisons:\n")
        rand_index_pred = 5
        random_numbers = [random.randint(0, actual_labels.shape[0] - 1) for _ in range(5)]
        for i in random_numbers:
            f.write(f"For i = {i}, we have:\n")
            f.write(f"Y_predictions[i]     = {predictions[i]}\n")
            f.write(f"Y_test_normalized[i] = {actual_labels[i]}\n\n")

    print(f"All results saved to {filenameFull}!")
    return filenameFull

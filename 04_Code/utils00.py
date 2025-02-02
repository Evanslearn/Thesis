import os
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
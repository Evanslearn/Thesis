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

def doTrainValTestSplit(X_data, Y_targets, test_ratio, val_ratiofromTrainVal, random_state):
    val_ratio = val_ratiofromTrainVal

    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_data, Y_targets, test_size=test_ratio,
                                                                random_state=random_state)#, stratify=Y_targets)

    print(
        f'''We have already used {test_ratio * 100}% of the data for the test set. So now, the val_ratio = {val_ratio * 100}%
    of the val-train data, translates to {(1 - test_ratio) * val_ratio * 100} of the total data.''')
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=val_ratio,
                                                      random_state=random_state)#, stratify=Y_train_val)

    print(f'train - {Y_train}, \nval - {Y_val},\ntest {Y_test}')

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
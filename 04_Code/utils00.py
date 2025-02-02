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
    X_train, X_test_val, Y_train, Y_test_val = train_test_split(X_data, Y_targets, test_size=test_val_ratio,
                                                                random_state=random_state)  # , stratify=Y_targets)

    val_ratio = test_val_ratio * valRatio_fromTestVal
    print(
        f'''We have used {test_val_ratio * 100}% of the data for the test+val set. So now, the val_ratio = {valRatio_fromTestVal * 100}%
    of the val-test data, translates to {val_ratio * 100}% of the total data.''')
    X_test, X_val, Y_test, Y_val = train_test_split(X_test_val, Y_test_val, test_size=valRatio_fromTestVal,
                                                      random_state=random_state)#, stratify=Y_train_val)

    print(f'train - {Y_train}, \nval - {Y_val},\ntest {Y_test}')

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, val_ratio
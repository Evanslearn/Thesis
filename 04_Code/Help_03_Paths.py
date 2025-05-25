import os

# Define base paths
EMBEDDINGS_PATH = "/02_Embeddings/"
FOLDER_PATH = os.getcwd() + EMBEDDINGS_PATH

# List of all file paths
DATA_FILES = [
    "Embeddings_Lu_2025-01-15_23-11-50.csv",
    "Embeddings_Pitt_2025-01-21_02-02-38.csv",
    "Embeddings_Pitt_2025-01-26_23-29-29.csv",
    "Embeddings_Pitt_2025-01-28_00-39-51.csv",
    "Embeddings_Pitt_2025-01-29_22-14-49.csv",
    "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-01-30_00-49-18.csv",
    "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-02_16-27-13.csv",
    "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-02_23-37-08.csv",
    "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-22_15-09-08.csv",
    "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-24_00-04-20.csv",
    "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-25_21-15-00.csv",
    "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-02.csv",
    "Embeddings_Pitt_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv",
    "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv",
    "Embeddings_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-06.csv",
    "Embeddings__Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_18-05-52.csv",
    "Embeddings__Pitt__nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_23-38-19.csv"
]
LABEL_FILES = [
    "Lu_sR50_2025-01-06_01-40-21_output.csv",
    "Labels_Pitt_2025-01-21_02-05-52.csv",
    "Labels_Pitt_2025-01-26_23-29-29.csv",
    "Labels_Pitt_2025-01-30_00-49-18.csv",
    "Labels_Pitt_2025-02-02_16-27-13.csv",
    "Labels_Pitt_2025-02-02_23-37-08.csv",
    "Labels_Pitt_2025-02-20_20-22-02.csv",
    "Labels_Pitt_2025-02-22_15-09-08.csv",
    "Labels_Pitt_2025-02-24_00-04-20.csv",
    "Labels_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-25_21-15-00.csv",
    "Labels_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-02.csv",
    "Labels_Pitt_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv",
    "Labels_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv",
    "Labels_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-06.csv",
    "Labels__Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_18-05-52.csv",
    "Labels__Pitt__nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_23-38-19.csv"
]
INDICES_FILES = [
    "Indices_Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-06.csv",
    "Indices__Pitt_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_18-05-52.csv",
    "Indices__Pitt__nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_23-38-19.csv"
]

# Data File Paths
DATA_FILES_TRAIN = [
    "Embeddings_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv",
    "Embeddings_Pitt_trainSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv",
    "Embeddings_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv",
    "Embeddings_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-06.csv",
    "Embeddings__Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_18-05-52.csv",
    "Embeddings__Pitt__trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_23-38-19.csv"
]
DATA_FILES_VAL = [
    "Embeddings_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv",
    "Embeddings_Pitt_valSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv",
    "Embeddings_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv",
    "Embeddings_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-07.csv",
    "Embeddings__Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_18-05-52.csv",
    "Embeddings__Pitt__valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_23-38-20.csv"
]
DATA_FILES_TEST = [
    "Embeddings_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv",
    "Embeddings_Pitt_testSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv",
    "Embeddings_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv",
    "Embeddings_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-07.csv",
    "Embeddings__Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_18-05-52.csv",
    "Embeddings__Pitt__testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_23-38-20.csv"
]

# Label File Paths
LABEL_FILES_TRAIN = [
    "Labels_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv",
    "Labels_Pitt_trainSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv",
    "Labels_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv",
    "Labels_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-06.csv",
    "Labels__Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_18-05-52.csv",
    "Labels__Pitt__trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_23-38-19.csv"
]
LABEL_FILES_VAL = [
    "Labels_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv",
    "Labels_Pitt_valSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv",
    "Labels_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv",
    "Labels_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-07.csv",
    "Labels__Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_18-05-52.csv",
    "Labels__Pitt__valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_23-38-20.csv"
]
LABEL_FILES_TEST = [
    "Labels_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_01-02-03.csv",
    "Labels_Pitt_testSet_nCl2_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-02-26_22-25-37.csv",
    "Labels_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-02_01-14-29.csv",
    "Labels_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-07.csv",
    "Labels__Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_18-05-52.csv",
    "Labels__Pitt__testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_23-38-20.csv"
]

INDICES_FILES_TRAIN = [
    "Indices_Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-06.csv",
    "Indices__Pitt_trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_18-05-52.csv",
    "Indices__Pitt__trainSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_23-38-19.csv"
]
INDICES_FILES_VAL = [
    "Indices_Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-07.csv",
    "Indices__Pitt_valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_18-05-52.csv",
    "Indices__Pitt__valSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_23-38-20.csv"
]
INDICES_FILES_TEST = [
    "Indices_Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-09_21-57-07.csv",
    "Indices__Pitt_testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_18-05-52.csv",
    "Indices__Pitt__testSet_nCl5_nN50_winSize10_stride1_winSizeSkip20_nEmbeddings300_2025-03-16_23-38-20.csv"
]


def returnFileNames(search_dir, caseType, common_part):
    import glob

    print(search_dir)
    # Get all CSV files in the directory
    csv_files = glob.glob(f"{search_dir}/*.csv")

    # Filter files that contain the common part dynamically
    matching_files = {f: f for f in csv_files if common_part in f}
    matching_files = [f for f in csv_files if common_part in f]
 #   print("All CSV files:", csv_files)
 #   print("Filtered Matching Files:", matching_files)

    data_train = next((f for f in matching_files if f"Embeddings__{caseType}_trainSet" in f), None)
    data_val = next((f for f in matching_files if f"Embeddings__{caseType}_valSet" in f), None)
    data_test = next((f for f in matching_files if f"Embeddings__{caseType}_testSet" in f), None)
    data_general = next((f for f in matching_files if
                         f"Embeddings__{caseType}_" in f and all(x not in f for x in ["trainSet", "valSet", "testSet"])),
                        None)

    # Categorize files based on their naming pattern
    labels_train = next((f for f in matching_files if f"Labels__{caseType}_trainSet" in f), None)
    labels_val = next((f for f in matching_files if f"Labels__{caseType}_valSet" in f), None)
    labels_test = next((f for f in matching_files if f"Labels__{caseType}_testSet" in f), None)
    labels_general = next((f for f in matching_files if
                           f"Labels__{caseType}_" in f and all(x not in f for x in ["trainSet", "valSet", "testSet"])),
                          None)

    indices_train = next((f for f in matching_files if f"Indices__{caseType}_trainSet" in f), None)
    indices_val = next((f for f in matching_files if f"Indices__{caseType}_valSet" in f), None)
    indices_test = next((f for f in matching_files if f"Indices__{caseType}_testSet" in f), None)
    indices_general = next((f for f in matching_files if f"Indices__{caseType}_" in f and all(
        x not in f for x in ["trainSet", "valSet", "testSet"])), None)

    # Print results
 #   print(f"Data Train: {data_train}")
 #   print(f"Data Validation: {data_val}")
 #   print(f"Data Test: {data_test}")
 #   print(f"Data General: {data_general}")  # The 4th case

 #   print(f"Labels Train: {labels_train}")
 #   print(f"Labels Validation: {labels_val}")
 #   print(f"Labels Test: {labels_test}")
 #   print(f"Labels General: {labels_general}")  # The 4th case

 #   print(f"Indices Train: {indices_train}")
 #   print(f"Indices Validation: {indices_val}")
 #   print(f"Indices Test: {indices_test}")
 #   print(f"Indices General: {indices_general}")  # The 4th case

    # Return all 12 variables
    return (labels_train, labels_val, labels_test, labels_general,
            indices_train, indices_val, indices_test, indices_general,
            data_train, data_val, data_test, data_general)


# Define the common part of the filenames
#common_part = "scalerMinMax_sil0.5099_nCl150_nN15_wSize2_str1024_wSizeSG6_SGLossNCE_nEmbd200_2025-04-20_20-15-11"
common_part = "scalerMinMax_sil0.6531_nCl100_nN50_wSize128_str128_wSizeSG10_SGLossNCE_nEmbd300_2025-04-21_00-33-13"
common_part = "scalerStandard_sil0.6895_nCl6_nN15_wSize64_str2048_wSizeSG6_SGLossNCE_nEmbd100_2025-04-21_22-02-16"
common_part = "scalerMinMax_sil0.4427_nCl6_nN5_wSize4096_str2048_wSizeSG6_SGLossNCE_nEmbd100_2025-04-22_19-52-30"
common_part = 'scalerMinMax_sil0.4472_nCl6_nN5_wSize4096_str2048_wSizeSG6_SGLossNCE_nEmbd100_2025-04-24_02-41-33'
common_part = "scalerMinMax_sil0.3961_nCl6_nN5_wSize4096_str2048_wSizeSG6_SGLossNCE_nEmbd300_2025-04-24_21-48-39"
common_part = "scalerMinMax_sil0.4367_nCl6_nN5_wSize1024_str512_wSizeSG6_SGLossNCE_nEmbd200_2025-04-26_16-35-37"
#common_part = "scalerMinMax_sil0.3961_nCl6_nN5_wSize4096_str2048_wSizeSG6_SGLossNCE_nEmbd100_2025-04-25_23-41-00"
#common_part = "scalerMinMax_sil0.3961_nCl6_nN5_wSize4096_str2048_wSizeSG6_SGLossNCE_nEmbd200_2025-04-26_02-55-48"
common_part = "scalerMinMax_sil0.3961_nCl6_nN5_wSize4096_str2048_wSizeSG6_SGLossNCE_nEmbd300_2025-04-24_21-48-39"
common_part = "scalerMinMax_sil0.4367_nCl6_nN5_wSize1024_str512_wSizeSG6_SGLossNCE_nEmbd200_2025-04-27_21-31-35"
common_part = "scalerMinMax_sil0.3839_nCl6_nN4_wSize4_str2_wSizeSG6_SGLossNCE_nEmbd200_2025-05-03_00-09-27"
common_part = "scalerMinMax_sil0.4549_nCl6_nN4_wSize8_str4_wSizeSG6_SGLossNCE_nEmbd200_2025-05-03_00-57-09"
common_part = "scalerMinMax_sil0.3625_nCl8_nN4_wSize8_str4_wSizeSG6_SGLossNCE_nEmbd200_2025-05-03_01-31-55"
common_part = "scalerStandard_sil0.5108_nCl6_nN4_wSize8_str4_wSizeSG6_SGLossNCE_nEmbd200_2025-05-03_16-17-43"
common_part = "scalerStandard_sil0.6477_nCl8_nN4_wSize8_str4_wSizeSG6_SGLossNCE_nEmbd5_2025-05-03_21-46-51"
common_part = "scalerStandard_sil0.6723_nCl16_nN4_wSize36_str9_wSizeSG6_SGLossNCE_nEmbd5_2025-05-04_01-03-39"
common_part = "scalerStandard_sil0.8626_nCl2_nN4_wSize2048_str1024_wSizeSG6_SGLossNCE_nEmbd50_2025-05-20_06-14-58"

search_dir = os.getcwd() + "/02_Embeddings"
# Get the returned file names
new_files = returnFileNames(search_dir, caseType="Pitt", common_part=common_part)
# Extract only filenames
new_files = tuple(os.path.basename(path) for path in new_files)

print(new_files)

LABEL_FILES_TRAIN.append(new_files[0])
LABEL_FILES_VAL.append(new_files[1])
LABEL_FILES_TEST.append(new_files[2])
LABEL_FILES.append(new_files[3])

INDICES_FILES_TRAIN.append(new_files[4])
INDICES_FILES_VAL.append(new_files[5])
INDICES_FILES_TEST.append(new_files[6])
INDICES_FILES.append(new_files[7])

DATA_FILES_TRAIN.append(new_files[8])
DATA_FILES_VAL.append(new_files[9])
DATA_FILES_TEST.append(new_files[10])
DATA_FILES.append(new_files[11])


# Always use the last value (latest file)
FILEPATH_DATA = DATA_FILES[-1]
FILEPATH_DATA_TRAIN = DATA_FILES_TRAIN[-1]
FILEPATH_DATA_VAL = DATA_FILES_VAL[-1]
FILEPATH_DATA_TEST= DATA_FILES_TEST[-1]

FILEPATH_LABELS = LABEL_FILES[-1]
FILEPATH_LABELS_TRAIN = LABEL_FILES_TRAIN[-1]
FILEPATH_LABELS_VAL = LABEL_FILES_VAL[-1]
FILEPATH_LABELS_TEST = LABEL_FILES_TEST[-1]

FILEPATH_INDICES = INDICES_FILES[-1]
FILEPATH_INDICES_TRAIN = INDICES_FILES_TRAIN[-1]
FILEPATH_INDICES_VAL = INDICES_FILES_VAL[-1]
FILEPATH_INDICES_TEST = INDICES_FILES_TEST[-1]

# Construct full paths
FULL_PATH_DATA = os.path.join(FOLDER_PATH, FILEPATH_DATA)
FULL_PATH_DATA_TRAIN = os.path.join(FOLDER_PATH, FILEPATH_DATA_TRAIN)
FULL_PATH_DATA_VAL = os.path.join(FOLDER_PATH, FILEPATH_DATA_VAL)
FULL_PATH_DATA_TEST = os.path.join(FOLDER_PATH, FILEPATH_DATA_TEST)

FULL_PATH_LABELS = os.path.join(FOLDER_PATH, FILEPATH_LABELS)
FULL_PATH_LABELS_TRAIN = os.path.join(FOLDER_PATH, FILEPATH_LABELS_TRAIN)
FULL_PATH_LABELS_VAL = os.path.join(FOLDER_PATH, FILEPATH_LABELS_VAL)
FULL_PATH_LABELS_TEST = os.path.join(FOLDER_PATH, FILEPATH_LABELS_TEST)

FULL_PATH_INDICES = os.path.join(FOLDER_PATH, FILEPATH_INDICES)
FULL_PATH_INDICES_TRAIN = os.path.join(FOLDER_PATH, FILEPATH_INDICES_TRAIN)
FULL_PATH_INDICES_VAL = os.path.join(FOLDER_PATH, FILEPATH_INDICES_VAL)
FULL_PATH_INDICES_TEST = os.path.join(FOLDER_PATH, FILEPATH_INDICES_TEST)

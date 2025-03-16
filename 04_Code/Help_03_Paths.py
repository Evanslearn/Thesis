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

# STEP 1: Prepare your Data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

abspath = "/home/vang/Downloads/"
filepath_data = "Lu_sR50_2025-01-06_01-40-21_output (Copy).csv"
filepath_data = "Pitt_sR11025.0_2025-01-20_23-11-13_output.csv"
totalpath_data = abspath + filepath_data
data = pd.read_csv(totalpath_data, header=None)

filepath_labels = "Lu_sR50_2025-01-06_01-40-21_output.csv"
totalpath_labels = abspath + filepath_labels
initial_labels = pd.read_csv(totalpath_labels, header=None)[:][0]
print(type(initial_labels))
filepath_labels = "Pitt_sR11025.0_2025-01-20_23-12-07_labels.csv"
totalpath_labels = abspath + filepath_labels
initial_labels = pd.read_csv(totalpath_labels, header=None)
if type(initial_labels) != type(pd.Series):
    initial_labels = initial_labels.iloc[:, 0] # convert to series
print(type(initial_labels))


# Drop NaN rows from data
data = data.dropna()

# Reset indices after dropping rows
data = data.reset_index(drop=True)

# Ensure labels align with the updated data
labels = initial_labels[data.index]
labels = labels.reset_index(drop=True)



labels = labels.map({'C': 0, 'D': 1}).to_numpy()
print(labels)




def step1_normalization(normFlag):
    if normFlag == "initial":
        #print(data)
        # Apply Min-Max Scaling row-wise
        scaler = MinMaxScaler()
        scaled_data = data.apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), axis=1)
        # Convert back to DataFrame
        scaled_data = pd.DataFrame(scaled_data)
        #print(scaled_data)
    else:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

    print(f"data.shape = {data.shape} and type(data) = {type(data)}")
    print(f"scaled_data.shape = {scaled_data.shape} and type(scaled_data) = {type(scaled_data)}")
    print(data[0][0:5])
    print(scaled_data[0][0:5])
    return scaled_data
normFlag = "Test"
scaled_data = step1_normalization(normFlag)


# STEP 2: sliding window segmentation
def step2_SegmentingSlidingWindow():
    import numpy as np

    # Sliding window segmentation
    window_size = 30
    segments = [scaled_data.iloc[i:i+window_size, :].values.flatten() for i in range(len(scaled_data) - window_size)]
    print(type(segments[0][0][0]))

    # Check if all segments have the correct shape
    for idx, seg in enumerate(segments):
        print(f"Segment {idx} shape: {seg.shape} and type: {type(seg)}")

    # Convert segments to a 2D numpy array
    segments = np.array(segments)

    # Print the shape to confirm it is a 2D array
    print(f"Segments shape after conversion: {segments.shape}")  # Should be (num_segments, 30)

    #segments = np.vstack(segments)  # Vertically stack to get (num_segments, 30)
    segments = np.stack(segments)
    print(segments.shape)  # Should be (24, 30)

    # Check shape
    #p#rint(type(segments))
    #print(segments.shape)
    #print(segments[0])  # Print the first segment
    #print(segments[0].shape)
#step2_SegmentingSlidingWindow()


# STEP 3: Tokenization (K - MeansClustering)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report


def step3_KMeansTokenization_Simple():
    # INITIAL ATTEMPT WITH SEGMENTS - WAS"T WORKING
    #    segments = scaled_data
    #   kmeans = KMeans(n_clusters=50, random_state=0).fit(segments)
    #    tokens = kmeans.predict(segments)  # MAYBE USE THIS, AND NOT LABELS, TO GENERALIZE TO VAl/TEST DATA?

    # Define the number of clusters
    k = 6  # Replace with your desired number of
    random_state = 0

    # Initialize and fit k-means
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(scaled_data)

    # Get cluster labels
    cluster_labels = kmeans.labels_

    # Get cluster centroids
    centroids = kmeans.cluster_centers_
    return cluster_labels, centroids, kmeans

def step3_KMeansTokenization():
    sil_scores = []
    cluster_range = range(2, 11)  # Try clustering with 2 to 10 clusters
    random_state = 0

    # Step 3: Try different numbers of clusters and select the one with the highest Silhouette Score
    sil_scores = []
    best_model = None
    best_n_clusters = 0
    best_sil_score = -1  # Initialize with a very low score
    best_sil_labels = []
    best_sil_centroids = []

    cluster_range = range(2, 11)  # Try clustering with 2 to 10 clusters

    for n_clusters in cluster_range:
        # Apply K-means with the current number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(scaled_data)
        centroids = kmeans.cluster_centers_

        # Calculate the Silhouette Score
        sil_score = silhouette_score(scaled_data, cluster_labels)
        sil_scores.append(sil_score)

        # Track the best model (highest Silhouette Score)
        if sil_score > best_sil_score:
            best_sil_labels = cluster_labels
            best_sil_centroids = centroids
            best_sil_score = sil_score
            best_n_clusters = n_clusters
            best_model = kmeans

        print(f"Silhouette Score for {n_clusters} clusters: {sil_score:.3f}")

    print(F"Best Silhouette Score = {best_sil_score} for clusters = {best_n_clusters}")
    print("REMINDER TO USE CRITERION WITH SILHOUETTE FOR SIGANAL2VEC")

    return best_sil_labels, best_sil_centroids, best_model
#labels, centroids, kmeans = step3_KMeansTokenization_Simple()
cluster_labels, centroids, kmeans = step3_KMeansTokenization()

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def step3b_KMeansPlot(labels, centroids, kmeans):
    print(labels)
    plt.figure()
    for i, centroid in enumerate(kmeans.cluster_centers_):
        plt.plot(centroid, label=f'Cluster {i}')
    plt.legend()
    plt.title('Cluster Centroids')
    plt.pause(0.001)

    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Scatter plot
    plt.figure()
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.title('Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.pause(0.001)
    plt.show()
#step3b_KMeansPlot(labels, centroids, kmeans)

# STEP 4: Apply Skip-gram Model (Training Embeddings)
from gensim.models import Word2Vec
def step4_NGRAM(cluster_labels, vector_size):
    flagnGRAM = "NO"
    # Convert all labels to strings
    cluster_labels = [str(label) for label in cluster_labels]

    # Prepare Data for Skip-Gram Model (Word2Vec)
    context_pairs = []

    if flagnGRAM == "CLUSTER":
        # For simplicity, using a window size of 1 for the Skip-Gram model
        for i in range(1, len(cluster_labels) - 1):
            target = cluster_labels[i]  # Current segment as the target
            context = [cluster_labels[i - 1], cluster_labels[i + 1]]  # Previous and next segments as context
            for ctx in context:
                context_pairs.append([target, ctx])
    else:
        for i, target in enumerate(cluster_labels):
            for j, context in enumerate(cluster_labels):
                if i != j:  # Avoid self-pairs
                    context_pairs.append([target, context])


    # Train Word2Vec Model (Skip-Gram)
    model = Word2Vec(sentences=context_pairs, vector_size=vector_size, window=5, sg=1, min_count=1)

    if flagnGRAM == "CLUSTER":
        # Obtain Embeddings for All Segments
        embeddings = np.array([model.wv[label] for label in cluster_labels])
    else:
        # Obtain Embeddings for All Cluster Labels
        unique_labels = list(set(cluster_labels))
        embeddings = np.array([model.wv[label] for label in unique_labels])

    return embeddings, model
embeddings, model_NGRAM = step4_NGRAM(cluster_labels, vector_size=data.shape[1])
#print(embeddings)
print(type(embeddings))
print(embeddings.shape)

# Assume:
# - embeddings: np.array of shape (3, 100) (one embedding per cluster label)
# - cluster_labels: list/array of length 54, with the cluster label for each time series

# Map each time series to its corresponding cluster embedding
time_series_embeddings = np.array([embeddings[label] for label in cluster_labels])

# Now, time_series_embeddings will have shape (54, 100)
print(f"TS embeddings.shape = {time_series_embeddings.shape}")  # Output: (54, 100)


def step5_KNN(embeddings, labels):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    # Initialize k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Step 9: Evaluate k-NN Classifier
    y_pred = knn.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))
step5_KNN(time_series_embeddings, labels)

def step6_VisualizeFinal():
    # Step 10: Optional Visualization (PCA to reduce embeddings to 2D)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(time_series_embeddings)

    # Plot the embeddings
    plt.figure()
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.pause(0.001)
    plt.show()
step6_VisualizeFinal()

import time
from datetime import datetime
def step7_SaveEmbeddingsToOutput(embeddings):
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    df = pd.DataFrame(embeddings)
    filename = abspath + "Embeddings" + "_" + formatted_datetime + ".csv"

    # Writing to CSV with pandas (which is generally faster)
    df.to_csv(filename, index=False, header=False)


    df_labels = pd.DataFrame(labels)
    filename = abspath + "Labels" + "_" + formatted_datetime + ".csv"
    df_labels.to_csv(filename, index=False, header=False)
    pass;

step7_SaveEmbeddingsToOutput(time_series_embeddings)
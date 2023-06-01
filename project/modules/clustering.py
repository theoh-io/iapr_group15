import numpy as np

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def cluster_features(features, max_clusters, use_dbscan=False, dbscan_eps=0.5, dbscan_min_samples=5):
    """
    Cluster the given features using K-means or DBSCAN algorithm with the optimal number of clusters determined by the Silhouette Score.
    
    Parameters:
        - features: Feature matrix with shape (num_samples, num_features).
        - max_clusters: Maximum number of clusters to consider.
        - use_dbscan: Flag to indicate whether to use DBSCAN algorithm for clustering.
        - dbscan_eps: The maximum distance between two samples for them to be considered as neighbors in DBSCAN.
        - dbscan_min_samples: The minimum number of samples in a neighborhood for a point to be considered as a core point in DBSCAN.
    
    Returns:
        - labels: Cluster labels for each sample.
    """
    
    silhouette_scores = []
    
    if use_dbscan:
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = dbscan.fit_predict(features)
    else:
        # Perform K-means clustering for different number of clusters
        for num_clusters in range(2, max_clusters+1):
            kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(features)
            silhouette_scores.append(silhouette_score(features, labels))
    
        # Determine the optimal number of clusters based on the Silhouette Score
        optimal_num_clusters = 2 + silhouette_scores.index(max(silhouette_scores))
        
        # Perform K-means clustering with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_num_clusters, n_init=10, random_state=42)
        labels = kmeans.fit(features).labels_
        labels = outliers_k_means(labels)
    
    return labels


def clusters_check(features, labels):
    labels = check_outlier_cluster(labels, max_num_outliers=3)
    labels = combine_clusters(labels, acceptable_sizes=(9, 12, 16))
    labels = separate_mixed_cluster(features, labels, acceptable_sizes=(9, 12, 16))
    return labels


def outliers_k_means(labels, num_outliers_max=3):
    # Outlier detection based on smaller cluster size
    # Count the occurrences of each cluster
    cluster_counts = np.bincount(labels)
    
    # Find the index of the cluster with the least occurrences
    if cluster_counts[np.argmin(cluster_counts)] <= num_outliers_max: 
        outlier_cluster = np.argmin(cluster_counts)
        # Replace outlier label by -1
        labels[labels == outlier_cluster] = -1
    else:
        outlier_cluster = None
    
    return labels


def remove_false_outliers(labels, max_num_outliers=3):
    # Count the number of outliers in the outlier cluster
    outlier_indices = np.where(labels == -1)[0]
    num_outliers = outlier_indices.shape[0]

    # If the number of outliers exceeds the maximum, remove the outlier cluster
    if num_outliers > max_num_outliers:
        # Create a new cluster for the removed outlier points
        new_cluster_index = np.max(labels) + 1
        labels[outlier_indices] = new_cluster_index
    
    return labels


def detect_outlier_cluster(labels, acceptable_sizes=(9,12,16), max_num_outliers=3):
    # Check that there is not an outlier cluster
    unique_labels = np.unique(labels)
    if -1 not in unique_labels:
        # Get the size of the clusters
        cluster_sizes=[]
        for label in unique_labels:
            cluster_pts_idx = np.where(labels == label)[0]
            cluster_size = cluster_pts_idx.shape[0]
            cluster_sizes.append(cluster_size)
        
        # Check if there is only one cluster with size not in acceptable_size
        # Count the number of clusters with acceptable sizes
        count_acceptable_sizes = sum(size in acceptable_sizes for size in cluster_sizes)
        
        if count_acceptable_sizes == len(cluster_sizes) - 1:
            print("There is only one cluster that doesn't have the acceptable size.")
            # Find the index of the cluster that doesn't have the acceptable size
            outlier_cluster_index=[index for index, size in enumerate(cluster_sizes) if size not in acceptable_sizes]
            
            # Find the index of the outlier cluster in the labels array
            outlier_cluster_label = unique_labels[outlier_cluster_index]
            
            #replace the labels to -1 for the outlier cluster
            labels[labels==outlier_cluster_label]=-1
        else:
            print("There are multiple clusters that don't have the acceptable size.")
    
    return labels


def check_outlier_cluster(labels, max_num_outliers=3):
    remove_false_outliers(labels)
    detect_outlier_cluster(labels)
    return labels


def combine_clusters(labels, acceptable_sizes=(9, 12, 16)):
    # Assemble the clusters with acceptable sizes
    unique_labels = np.unique(labels)
    
    # Remove the outlier cluster
    unique_labels = unique_labels[unique_labels!=-1]
    
    # Iterate over the clusters
    for i, label in enumerate(unique_labels):
        # Get the points of the cluster
        cluster_pts_idx = np.where(labels == label)[0]

        cluster_size = cluster_pts_idx.shape[0]
        if cluster_size < min(acceptable_sizes):
            # Check if combining with another cluster can reach an acceptable size
            for j, label2 in enumerate(np.unique(labels)):
                if j == i:
                    # Skip the current cluster
                    continue
                else:
                    other_cluster_pts_idx = np.where(labels == label2)[0]
                    other_cluster_size = other_cluster_pts_idx.shape[0]
                    combined_size = cluster_size + other_cluster_size
                    if combined_size in acceptable_sizes:
                        # Combine the clusters
                        labels[other_cluster_pts_idx] = label
    
    return labels


def separate_mixed_cluster(features, labels, acceptable_sizes=(9, 12, 16)):
    # Separate a cluster mixing 2 and therefore having size acceptable_size*2
    
    # Get all possible combinations sizes of acceptable sizes
    combinations_sizes = []
    for i in acceptable_sizes:
        for j in acceptable_sizes:
            combinations_sizes.append(i+j)
    
    combinations_sizes = np.unique(combinations_sizes)
    unique_labels = np.unique(labels)
    
    # Remove the outlier cluster
    unique_labels=unique_labels[unique_labels!=-1]
    
    # Iterate over the clusters
    for i, label in enumerate(np.unique(labels)):
        # Check if cluster size is in combinations_sizes
        cluster_pts_idx = np.where(labels == label)[0]
        cluster_size = cluster_pts_idx.shape[0]
        
        if cluster_size in combinations_sizes:
            print(f"Cluster {i} has size {cluster_size} and is a mix of 2 clusters")
            cluster_pts = features[cluster_pts_idx]
            
            # Use kmeans to separate the cluster into 2 clusters
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            labels2 = kmeans.fit_predict(cluster_pts)
            
            # Modify labels 2 so that we create a new cluster label
            labels2=labels2+np.max(labels)+1
            
            # Update labels 
            labels[cluster_pts_idx]=labels2
            print(f"Updated labels: {labels}")
    return labels
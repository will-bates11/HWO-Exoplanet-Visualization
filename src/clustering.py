from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import logging

logger = logging.getLogger(__name__)

def determine_optimal_clusters(data, max_clusters=10):
    """
    Determine the optimal number of clusters using the silhouette score.
    
    Parameters:
    -----------
    data : array-like
        The data to cluster
    max_clusters : int
        Maximum number of clusters to try
    
    Returns:
    --------
    int
        Optimal number of clusters
    """
    if len(data) < max_clusters:
        return min(len(data), 3)
        
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    return np.argmax(silhouette_scores) + 2

def cluster_exoplanets(exoplanet_data, num_clusters=None):
    """
    Cluster exoplanets based on their physical characteristics.
    
    Parameters:
    -----------
    exoplanet_data : pandas.DataFrame
        DataFrame containing exoplanet data
    num_clusters : int, optional
        Number of clusters to use. If None, determined automatically.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added cluster assignments
    """
    try:
        # Select features for clustering
        features = ['pl_rade', 'st_dist', 'pl_orbper', 'habitability_index']
        
        # Create feature matrix, dropping rows with missing values
        X = exoplanet_data[features].dropna()
        
        if len(X) < 2:
            logger.warning("Not enough valid data points for clustering")
            exoplanet_data['cluster'] = 0
            return exoplanet_data
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters if not specified
        if num_clusters is None:
            num_clusters = determine_optimal_clusters(X_scaled)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        
        # Store original index to map clusters back to full dataset
        original_index = X.index
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Initialize all clusters as -1 (indicating no cluster)
        exoplanet_data['cluster'] = -1
        
        # Assign clusters only to rows that were used in clustering
        exoplanet_data.loc[original_index, 'cluster'] = cluster_labels
        
        # Add cluster centers and distances for visualization
        cluster_centers_scaled = kmeans.cluster_centers_
        cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
        
        # Store cluster centers as additional data
        exoplanet_data['cluster_center_dist'] = -1
        for i, idx in enumerate(original_index):
            cluster = cluster_labels[i]
            point = X_scaled[i]
            center = cluster_centers_scaled[cluster]
            distance = np.linalg.norm(point - center)
            exoplanet_data.loc[idx, 'cluster_center_dist'] = distance
        
        return exoplanet_data
        
    except Exception as e:
        logger.error(f"Error in clustering: {e}")
        # Fallback: assign all to same cluster
        exoplanet_data['cluster'] = 0
        return exoplanet_data

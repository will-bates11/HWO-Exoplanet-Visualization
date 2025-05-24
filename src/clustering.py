from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def determine_optimal_clusters(data: np.ndarray, max_clusters: int = 10) -> int:
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
    n_samples = len(data)
    
    # Need at least 2 samples per cluster
    max_feasible_clusters = min(max_clusters, n_samples // 2)
    
    if max_feasible_clusters < 2:
        return 1
    
    if n_samples < 4:  # Need at least 4 samples for meaningful clustering
        return min(2, max_feasible_clusters)
        
    silhouette_scores = []
    
    try:
        for n_clusters in range(2, max_feasible_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            # Check if all clusters have at least one point
            if len(np.unique(cluster_labels)) == n_clusters:
                silhouette_avg = silhouette_score(data, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(-1)  # Invalid clustering
        
        if not silhouette_scores or max(silhouette_scores) < 0:
            return 2  # Default fallback
            
        return np.argmax(silhouette_scores) + 2
        
    except Exception as e:
        logger.warning(f"Error in optimal cluster determination: {e}")
        return 3  # Safe default

def cluster_exoplanets(exoplanet_data: pd.DataFrame, num_clusters: Optional[int] = None) -> pd.DataFrame:
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

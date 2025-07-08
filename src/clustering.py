from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Dict, Any
from functools import lru_cache
import hashlib
import pickle
import os

logger = logging.getLogger(__name__)

# Cache directory for clustering results
CLUSTER_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "clustering")

class ClusterCache:
    """Caching system for clustering results to avoid recomputation."""
    
    def __init__(self):
        os.makedirs(CLUSTER_CACHE_DIR, exist_ok=True)
    
    def _get_data_hash(self, data: np.ndarray) -> str:
        """Generate hash for data array."""
        return hashlib.md5(data.tobytes()).hexdigest()
    
    def get_cached_result(self, data: np.ndarray, num_clusters: Optional[int] = None) -> Optional[np.ndarray]:
        """Get cached clustering result if available."""
        try:
            data_hash = self._get_data_hash(data)
            cache_key = f"cluster_{data_hash}_{num_clusters or 'auto'}.pkl"
            cache_file = os.path.join(CLUSTER_CACHE_DIR, cache_key)
            
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading clustering cache: {e}")
        return None
    
    def save_result(self, data: np.ndarray, result: np.ndarray, num_clusters: Optional[int] = None):
        """Save clustering result to cache."""
        try:
            data_hash = self._get_data_hash(data)
            cache_key = f"cluster_{data_hash}_{num_clusters or 'auto'}.pkl"
            cache_file = os.path.join(CLUSTER_CACHE_DIR, cache_key)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Error saving clustering cache: {e}")

# Global cache instance
cluster_cache = ClusterCache()

def preprocess_clustering_data(data: pd.DataFrame) -> Tuple[np.ndarray, pd.Index, StandardScaler]:
    """Optimized preprocessing for clustering data."""
    # Select features for clustering with better feature engineering
    base_features = ['pl_rade', 'st_dist', 'pl_orbper', 'habitability_index']
    
    # Add derived features for better clustering
    derived_features = []
    if 'st_teff' in data.columns:
        derived_features.append('st_teff')
    if 'pl_eqt' in data.columns:
        derived_features.append('pl_eqt')
    
    all_features = base_features + derived_features
    available_features = [f for f in all_features if f in data.columns]
    
    # Create feature matrix, dropping rows with missing values
    X = data[available_features].dropna()
    
    if len(X) < 2:
        raise ValueError("Not enough valid data points for clustering")
    
    # Apply log transformation to skewed features for better clustering
    log_features = ['st_dist', 'pl_orbper']
    for feature in log_features:
        if feature in X.columns:
            X[feature] = np.log1p(X[feature])  # log1p handles zero values better
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, X.index, scaler

@lru_cache(maxsize=128)
def determine_optimal_clusters_fast(data_shape: Tuple[int, int], data_hash: str, max_clusters: int = 8) -> int:
    """
    Fast optimal cluster determination with caching based on data characteristics.
    """
    n_samples = data_shape[0]
    
    # Use heuristics for very small datasets
    if n_samples < 10:
        return min(2, n_samples // 2) if n_samples >= 4 else 1
    
    # For larger datasets, use a more efficient approach
    if n_samples > 1000:
        # Use sample for large datasets to speed up computation
        max_clusters = min(max_clusters, 6)  # Limit for performance
    
    # Cap clusters based on dataset size
    max_feasible_clusters = min(max_clusters, n_samples // 3)
    
    if max_feasible_clusters < 2:
        return 1
    
    # Use simple heuristics for medium datasets
    if n_samples <= 50:
        return min(4, max_feasible_clusters)
    elif n_samples <= 200:
        return min(5, max_feasible_clusters)
    else:
        return min(6, max_feasible_clusters)

def determine_optimal_clusters_silhouette(data: np.ndarray, max_clusters: int = 10) -> int:
    """
    Determine optimal clusters using silhouette analysis with early stopping.
    """
    n_samples = len(data)
    
    # Quick heuristics for small datasets
    if n_samples < 10:
        return min(3, n_samples // 2) if n_samples >= 6 else 2
    
    max_feasible_clusters = min(max_clusters, n_samples // 3)
    
    if max_feasible_clusters < 2:
        return 1
    
    best_score = -1
    best_k = 2
    scores = []
    
    try:
        # Use MiniBatchKMeans for faster computation on larger datasets
        clusterer = MiniBatchKMeans if n_samples > 500 else KMeans
        
        for k in range(2, min(max_feasible_clusters + 1, 8)):  # Limit range for performance
            if clusterer == MiniBatchKMeans:
                kmeans = clusterer(n_clusters=k, random_state=42, batch_size=min(1000, n_samples // 2))
            else:
                kmeans = clusterer(n_clusters=k, random_state=42, n_init=5)  # Reduced n_init
            
            cluster_labels = kmeans.fit_predict(data)
            
            # Check if clustering is valid
            if len(np.unique(cluster_labels)) == k:
                score = silhouette_score(data, cluster_labels)
                scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                
                # Early stopping if score starts decreasing significantly
                if len(scores) > 2 and scores[-1] < scores[-2] * 0.95:
                    break
            else:
                scores.append(-1)
        
        return best_k if best_score > 0 else 3
        
    except Exception as e:
        logger.warning(f"Error in silhouette analysis: {e}")
        return min(3, max_feasible_clusters)

def cluster_exoplanets_fast(exoplanet_data: pd.DataFrame, num_clusters: Optional[int] = None, 
                           use_cache: bool = True) -> pd.DataFrame:
    """
    Fast exoplanet clustering with optimizations and caching.
    
    Parameters:
    -----------
    exoplanet_data : pd.DataFrame
        DataFrame containing exoplanet data
    num_clusters : int, optional
        Number of clusters to use. If None, determined automatically.
    use_cache : bool
        Whether to use caching for clustering results
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added cluster assignments
    """
    try:
        # Preprocess data
        X_scaled, original_index, scaler = preprocess_clustering_data(exoplanet_data)
        
        if len(X_scaled) < 2:
            logger.warning("Not enough valid data points for clustering")
            exoplanet_data['cluster'] = 0
            return exoplanet_data
        
        # Check cache first
        cached_result = None
        if use_cache:
            cached_result = cluster_cache.get_cached_result(X_scaled, num_clusters)
        
        if cached_result is not None:
            cluster_labels = cached_result
            logger.info("Using cached clustering result")
        else:
            # Determine optimal number of clusters
            if num_clusters is None:
                data_hash = hashlib.md5(X_scaled.tobytes()).hexdigest()[:8]
                
                # Use fast heuristic for large datasets, detailed analysis for smaller ones
                if len(X_scaled) > 300:
                    num_clusters = determine_optimal_clusters_fast(X_scaled.shape, data_hash)
                else:
                    num_clusters = determine_optimal_clusters_silhouette(X_scaled)
            
            # Choose clustering algorithm based on dataset size
            if len(X_scaled) > 1000:
                # Use MiniBatchKMeans for large datasets
                kmeans = MiniBatchKMeans(
                    n_clusters=num_clusters, 
                    random_state=42,
                    batch_size=min(1000, len(X_scaled) // 2),
                    max_iter=100
                )
            else:
                # Use regular KMeans for smaller datasets
                kmeans = KMeans(
                    n_clusters=num_clusters, 
                    random_state=42,
                    n_init=10,
                    max_iter=300
                )
            
            # Perform clustering
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Cache the result
            if use_cache:
                cluster_cache.save_result(X_scaled, cluster_labels, num_clusters)
        
        # Initialize all clusters as -1 (indicating no cluster assigned)
        exoplanet_data['cluster'] = -1
        
        # Assign clusters only to rows that were used in clustering
        exoplanet_data.loc[original_index, 'cluster'] = cluster_labels
        
        # Add clustering quality metrics
        if len(np.unique(cluster_labels)) > 1:
            try:
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                logger.info(f"Clustering completed: {num_clusters} clusters, silhouette score: {silhouette_avg:.3f}")
            except Exception:
                pass
        
        return exoplanet_data
        
    except Exception as e:
        logger.error(f"Error in fast clustering: {e}")
        # Fallback: assign all to same cluster
        exoplanet_data['cluster'] = 0
        return exoplanet_data

# Maintain backward compatibility
def cluster_exoplanets(exoplanet_data: pd.DataFrame, num_clusters: Optional[int] = None) -> pd.DataFrame:
    """
    Backward compatible clustering function that uses the optimized implementation.
    """
    return cluster_exoplanets_fast(exoplanet_data, num_clusters, use_cache=True)

def get_cluster_summary(exoplanet_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for each cluster.
    """
    try:
        if 'cluster' not in exoplanet_data.columns:
            return {}
        
        summary = {}
        
        for cluster_id in exoplanet_data['cluster'].unique():
            if cluster_id == -1:  # Skip unassigned points
                continue
                
            cluster_data = exoplanet_data[exoplanet_data['cluster'] == cluster_id]
            
            summary[f"cluster_{cluster_id}"] = {
                'count': len(cluster_data),
                'avg_distance': float(cluster_data['st_dist'].mean()),
                'avg_radius': float(cluster_data['pl_rade'].mean()),
                'avg_habitability': float(cluster_data['habitability_index'].mean()),
                'distance_range': [float(cluster_data['st_dist'].min()), float(cluster_data['st_dist'].max())]
            }
        
        return summary
        
    except Exception as e:
        logger.warning(f"Error generating cluster summary: {e}")
        return {}

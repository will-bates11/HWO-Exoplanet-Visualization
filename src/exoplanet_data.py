import requests
import pandas as pd
import numpy as np
from utils import compute_habitability_index
import os
import json
import pickle
import gzip
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated NASA Exoplanet Archive API
NASA_EXOPLANET_API = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "exoplanet_cache.pkl.gz")
METADATA_FILE = os.path.join(CACHE_DIR, "cache_metadata.json")
CACHE_DURATION = timedelta(hours=6)  # Reduced cache duration for fresher data

# Pre-computed data indices for faster filtering
DISTANCE_INDEX_FILE = os.path.join(CACHE_DIR, "distance_index.pkl")

class DataCache:
    """Enhanced caching system with compression and indexing."""
    
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._memory_cache = {}
        self._max_memory_items = 10
    
    def _get_cache_key(self, query_params: Dict[str, Any]) -> str:
        """Generate cache key from query parameters."""
        return f"query_{hash(str(sorted(query_params.items())))}"
    
    def load_main_cache(self) -> Optional[pd.DataFrame]:
        """Load main cached exoplanet data with compression."""
        if not os.path.exists(CACHE_FILE) or not os.path.exists(METADATA_FILE):
            return None
        
        try:
            # Check metadata first
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
            
            cache_time = datetime.fromisoformat(metadata['timestamp'])
            if datetime.now() - cache_time > CACHE_DURATION:
                logger.info("Cache expired, will fetch fresh data")
                return None
            
            # Load compressed data
            with gzip.open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded {len(data)} records from compressed cache")
            return data
            
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
            return None
    
    def save_main_cache(self, data: pd.DataFrame) -> None:
        """Save exoplanet data with compression and indexing."""
        try:
            # Save compressed data
            with gzip.open(CACHE_FILE, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'record_count': len(data),
                'columns': list(data.columns),
                'data_hash': hash(str(data.values.tobytes()))
            }
            
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata, f)
            
            # Create distance index for faster filtering
            self._create_distance_index(data)
            
            logger.info(f"Cached {len(data)} records with compression")
            
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def _create_distance_index(self, data: pd.DataFrame) -> None:
        """Create pre-computed distance indices for faster filtering."""
        try:
            distance_indices = {}
            distance_ranges = [5, 10, 20, 30, 40, 50]
            
            for max_dist in distance_ranges:
                mask = data['st_dist'] <= max_dist
                distance_indices[max_dist] = data.index[mask].tolist()
            
            with open(DISTANCE_INDEX_FILE, 'wb') as f:
                pickle.dump(distance_indices, f)
                
        except Exception as e:
            logger.warning(f"Error creating distance index: {e}")
    
    def get_filtered_indices(self, max_distance: float) -> Optional[list]:
        """Get pre-computed indices for distance filtering."""
        try:
            if not os.path.exists(DISTANCE_INDEX_FILE):
                return None
                
            with open(DISTANCE_INDEX_FILE, 'rb') as f:
                distance_indices = pickle.load(f)
            
            # Find the closest pre-computed range
            available_ranges = sorted(distance_indices.keys())
            best_range = min(available_ranges, 
                           key=lambda x: abs(x - max_distance) if x >= max_distance else float('inf'))
            
            if best_range >= max_distance:
                return distance_indices[best_range]
                
        except Exception as e:
            logger.warning(f"Error loading distance index: {e}")
        
        return None

# Global cache instance
cache_manager = DataCache()

def validate_data_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Optimized vectorized data validation."""
    # Use vectorized operations for better performance
    initial_count = len(df)
    
    # Remove rows with missing essential data
    essential_cols = ['pl_name', 'st_dist']
    df = df.dropna(subset=essential_cols)
    
    # Vectorized numeric validation
    numeric_cols = ['st_dist', 'pl_rade', 'st_teff', 'pl_orbper']
    for col in numeric_cols:
        if col in df.columns:
            # Use vectorized operations
            df = df[df[col] > 0] if col in ['st_dist', 'pl_rade'] else df
    
    logger.info(f"Validated data: {len(df)}/{initial_count} records remaining")
    return df

def compute_habitability_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized habitability computation for better performance."""
    try:
        # Pre-allocate result array
        result = np.zeros(len(df))
        
        # Get required columns
        temp = df['st_teff'].fillna(0)
        radius = df['pl_rade'].fillna(0)
        period = df['pl_orbper'].fillna(0)
        
        # Vectorized calculations
        valid_mask = (temp > 0) & (radius > 0) & (period > 0)
        
        if valid_mask.any():
            # Temperature score (vectorized)
            temp_diff = np.abs(temp[valid_mask] - 5780)
            temp_score = np.exp(-temp_diff / 1000)
            
            # Size score (vectorized)
            size_score = np.exp(-(radius[valid_mask] - 1.0)**2 / 2)
            
            # Period score (vectorized)
            period_ratio = period[valid_mask] / 365.25
            period_score = np.exp(-(np.log10(period_ratio))**2 / 2)
            
            # Weighted combination
            final_scores = (0.5 * temp_score + 0.3 * size_score + 0.2 * period_score)
            result[valid_mask] = np.clip(final_scores, 0, 1)
        
        return pd.Series(result, index=df.index)
        
    except Exception as e:
        logger.warning(f"Error in vectorized habitability computation: {e}")
        # Fallback to individual computation
        return df.apply(compute_habitability_index, axis=1)

@lru_cache(maxsize=32)
def fetch_exoplanet_data_cached(max_distance: float = 50.0) -> pd.DataFrame:
    """Cached version with distance parameter for better cache utilization."""
    base_data = fetch_exoplanet_data()
    return base_data[base_data['st_dist'] <= max_distance].copy()

def fetch_exoplanet_data() -> pd.DataFrame:
    """
    Optimized exoplanet data fetching with enhanced caching and performance improvements.
    
    Returns:
    --------
    pandas.DataFrame
        Processed exoplanet data with computed habitability index
    """
    # Try to load from cache first
    cached_data = cache_manager.load_main_cache()
    if cached_data is not None:
        logger.info("Using cached exoplanet data")
        return cached_data

    logger.info("Fetching fresh exoplanet data from NASA API")
    
    try:
        # Optimized query with better column selection
        query = """
        select pl_name, sy_dist as st_dist, pl_rade, st_teff, pl_orbper, pl_bmassj, 
               st_mass, pl_eqt, pl_insol, pl_orbeccen, st_age
        from ps
        where sy_dist < 100 
        and pl_name is not null
        and sy_dist is not null
        order by sy_dist
        """
        
        params = {
            "query": query,
            "format": "json"
        }
        
        # Use session for connection pooling
        with requests.Session() as session:
            session.headers.update({
                'User-Agent': 'HWO-Exoplanet-Viz/1.0',
                'Accept-Encoding': 'gzip, deflate'
            })
            
            response = session.get(NASA_EXOPLANET_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        
        # Convert to DataFrame with optimized dtypes
        exoplanet_df = pd.DataFrame(data)
        logger.info(f"Fetched {len(exoplanet_df)} records from NASA API")
        
        # Optimize data types for memory efficiency
        numeric_columns = ['st_dist', 'pl_rade', 'st_teff', 'pl_orbper', 'pl_bmassj', 
                         'st_mass', 'pl_eqt', 'pl_insol', 'pl_orbeccen', 'st_age']
        
        for col in numeric_columns:
            if col in exoplanet_df.columns:
                exoplanet_df[col] = pd.to_numeric(exoplanet_df[col], errors='coerce', downcast='float')
        
        # Optimized validation
        exoplanet_df = validate_data_vectorized(exoplanet_df)
        
        # Vectorized habitability computation
        exoplanet_df['habitability_index'] = compute_habitability_vectorized(exoplanet_df)
        
        # Sort by distance for better cache locality
        exoplanet_df = exoplanet_df.sort_values('st_dist').reset_index(drop=True)
        
        # Cache the results
        cache_manager.save_main_cache(exoplanet_df)
        
        return exoplanet_df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from NASA API: {e}")
        # If we have cached data, use it as fallback even if expired
        cached_data = cache_manager.load_main_cache()
        if cached_data is not None:
            logger.warning("Using expired cache as fallback")
            return cached_data
        raise Exception("Failed to fetch exoplanet data and no cache available")

def get_filtered_data(max_distance: float) -> pd.DataFrame:
    """Optimized data filtering using pre-computed indices."""
    try:
        # Try to use pre-computed indices first
        base_data = fetch_exoplanet_data()
        indices = cache_manager.get_filtered_indices(max_distance)
        
        if indices is not None:
            # Use pre-computed indices
            filtered_indices = [i for i in indices if i < len(base_data)]
            return base_data.iloc[filtered_indices].copy()
        else:
            # Fallback to traditional filtering
            return base_data[base_data['st_dist'] <= max_distance].copy()
            
    except Exception as e:
        logger.error(f"Error in optimized filtering: {e}")
        # Fallback to basic filtering
        base_data = fetch_exoplanet_data()
        return base_data[base_data['st_dist'] <= max_distance].copy()

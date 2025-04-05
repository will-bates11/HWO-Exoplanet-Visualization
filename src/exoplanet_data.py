import requests
import pandas as pd
import numpy as np
from utils import compute_habitability_index
import os
import json
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated NASA Exoplanet Archive API
NASA_EXOPLANET_API = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "exoplanet_cache.json")
CACHE_DURATION = timedelta(days=1)  # Cache data for 1 day

def load_cache():
    """Load cached exoplanet data if it exists and is not expired."""
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        
        cache_time = datetime.fromisoformat(cache['timestamp'])
        if datetime.now() - cache_time > CACHE_DURATION:
            return None
            
        return pd.DataFrame(cache['data'])
    except Exception as e:
        logger.warning(f"Error loading cache: {e}")
        return None

def save_cache(data):
    """Save exoplanet data to cache."""
    try:
        cache = {
            'timestamp': datetime.now().isoformat(),
            'data': data.to_dict('records')
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        logger.warning(f"Error saving cache: {e}")

def fetch_exoplanet_data():
    """
    Fetch exoplanet data from NASA Exoplanet Archive with caching and error handling.
    
    Returns:
    --------
    pandas.DataFrame
        Processed exoplanet data with computed habitability index
    """
    # Try to load from cache first
    cached_data = load_cache()
    if cached_data is not None:
        logger.info("Using cached exoplanet data")
        return cached_data

    logger.info("Fetching fresh exoplanet data from NASA API")
    
    try:
        query = """
        select pl_name, sy_dist as st_dist, pl_rade, st_teff, pl_orbper, pl_bmassj, 
               st_mass, pl_eqt, pl_insol
        from ps
        where sy_dist < 50 
        and pl_name is not null
        """
        
        params = {
            "query": query,
            "format": "json"
        }
        
        response = requests.get(NASA_EXOPLANET_API, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        exoplanet_df = pd.DataFrame(data)
        
        # Convert columns to numeric, replacing invalid values with NaN
        numeric_columns = ['st_dist', 'pl_rade', 'st_teff', 'pl_orbper', 'pl_bmassj', 
                         'st_mass', 'pl_eqt', 'pl_insol']
        for col in numeric_columns:
            if col in exoplanet_df.columns:
                exoplanet_df[col] = pd.to_numeric(exoplanet_df[col], errors='coerce')
        
        # Compute habitability index
        exoplanet_df['habitability_index'] = exoplanet_df.apply(compute_habitability_index, axis=1)
        
        # Cache the results
        save_cache(exoplanet_df)
        
        return exoplanet_df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from NASA API: {e}")
        # If we have cached data, use it as fallback even if expired
        if cached_data is not None:
            logger.warning("Using expired cache as fallback")
            return cached_data
        raise Exception("Failed to fetch exoplanet data and no cache available")

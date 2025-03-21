import numpy as np
import pandas as pd

def compute_habitability_index(exoplanet):
    """
    Calculate a habitability index for an exoplanet based on its characteristics.
    
    Parameters:
    -----------
    exoplanet : dict or pandas.Series
        Dictionary containing exoplanet characteristics with keys:
        - 'st_teff': Stellar effective temperature (K)
        - 'pl_rade': Planet radius (Earth radii)
        - 'pl_orbper': Orbital period (days)
    
    Returns:
    --------
    float
        Habitability index between 0 and 1, where 1 is most Earth-like
    """
    # Handle missing values
    if pd.isna(exoplanet['st_teff']) or pd.isna(exoplanet['pl_rade']) or pd.isna(exoplanet['pl_orbper']):
        return 0.0
    
    # Temperature score (based on stellar temperature, weighted more heavily for Sun-like stars)
    temp_diff = abs(exoplanet['st_teff'] - 5780)  # Difference from Sun's temperature
    temp_score = np.exp(-temp_diff / 1000)  # Exponential decay for temperature difference
    
    # Size score (highest for Earth-like sizes, decreasing for larger planets)
    size_ratio = exoplanet['pl_rade']  # Ratio to Earth's radius
    size_score = np.exp(-(size_ratio - 1)**2 / 2)  # Gaussian function centered on Earth-like radius
    
    # Orbital period score (highest for Earth-like periods)
    period_ratio = exoplanet['pl_orbper'] / 365.25  # Ratio to Earth's orbital period
    period_score = np.exp(-(np.log10(period_ratio))**2 / 2)  # Log-scaled Gaussian for period
    
    # Weighted average of scores (temperature most important, then size, then orbit)
    weights = [0.5, 0.3, 0.2]
    final_score = (
        weights[0] * temp_score +
        weights[1] * size_score +
        weights[2] * period_score
    )
    
    return np.clip(final_score, 0, 1)  # Ensure score is between 0 and 1

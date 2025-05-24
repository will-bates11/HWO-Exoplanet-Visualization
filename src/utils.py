import numpy as np
import pandas as pd
from typing import Union

# Constants for habitability calculations
EARTH_TEMP = 5780  # Sun's effective temperature in Kelvin
EARTH_RADIUS = 1.0  # Earth radii
EARTH_PERIOD = 365.25  # Earth's orbital period in days

# Habitability weights (temperature, size, orbital period)
HABITABILITY_WEIGHTS = [0.5, 0.3, 0.2]

def compute_habitability_index(exoplanet: Union[dict, pd.Series]) -> float:
    """
    Calculate a habitability index for an exoplanet based on its characteristics.
    
    The habitability index is a composite score based on:
    - Stellar temperature (how Sun-like the host star is)
    - Planet radius (how Earth-like the planet size is)
    - Orbital period (how Earth-like the orbital period is)
    
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
    try:
        # Handle missing values
        required_fields = ['st_teff', 'pl_rade', 'pl_orbper']
        for field in required_fields:
            if field not in exoplanet or pd.isna(exoplanet[field]):
                return 0.0
        
        # Extract values with type checking
        stellar_temp = float(exoplanet['st_teff'])
        planet_radius = float(exoplanet['pl_rade'])
        orbital_period = float(exoplanet['pl_orbper'])
        
        # Validate ranges
        if stellar_temp <= 0 or planet_radius <= 0 or orbital_period <= 0:
            return 0.0
        
        # Temperature score (based on stellar temperature, weighted more heavily for Sun-like stars)
        temp_diff = abs(stellar_temp - EARTH_TEMP)
        temp_score = np.exp(-temp_diff / 1000)  # Exponential decay for temperature difference
        
        # Size score (highest for Earth-like sizes, decreasing for larger planets)
        size_score = np.exp(-(planet_radius - EARTH_RADIUS)**2 / 2)  # Gaussian function centered on Earth-like radius
        
        # Orbital period score (highest for Earth-like periods)
        period_ratio = orbital_period / EARTH_PERIOD
        # Use log scale to handle wide range of periods
        period_score = np.exp(-(np.log10(period_ratio))**2 / 2)  # Log-scaled Gaussian for period
        
        # Weighted average of scores
        final_score = (
            HABITABILITY_WEIGHTS[0] * temp_score +
            HABITABILITY_WEIGHTS[1] * size_score +
            HABITABILITY_WEIGHTS[2] * period_score
        )
        
        return np.clip(final_score, 0, 1)  # Ensure score is between 0 and 1
        
    except (ValueError, TypeError, KeyError) as e:
        # Return 0 for any calculation errors
        return 0.0

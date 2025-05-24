"""Tests for utils module."""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import compute_habitability_index


class TestHabitabilityIndex:
    """Test cases for habitability index calculation."""
    
    def test_earth_like_planet(self):
        """Test that Earth-like planet gets high habitability score."""
        earth_like = {
            'st_teff': 5780,  # Sun's temperature
            'pl_rade': 1.0,   # Earth radius
            'pl_orbper': 365.25  # Earth orbital period
        }
        score = compute_habitability_index(earth_like)
        assert 0.8 <= score <= 1.0, f"Earth-like planet should have high score, got {score}"
    
    def test_missing_data(self):
        """Test handling of missing data."""
        incomplete_data = {
            'st_teff': 5780,
            'pl_rade': np.nan,
            'pl_orbper': 365.25
        }
        score = compute_habitability_index(incomplete_data)
        assert score == 0.0, "Missing data should result in zero score"
    
    def test_invalid_values(self):
        """Test handling of invalid values."""
        invalid_data = {
            'st_teff': -1000,  # Negative temperature
            'pl_rade': 0,      # Zero radius
            'pl_orbper': -10   # Negative period
        }
        score = compute_habitability_index(invalid_data)
        assert score == 0.0, "Invalid values should result in zero score"
    
    def test_extreme_values(self):
        """Test handling of extreme but valid values."""
        extreme_data = {
            'st_teff': 10000,  # Very hot star
            'pl_rade': 10.0,   # Very large planet
            'pl_orbper': 10000  # Very long period
        }
        score = compute_habitability_index(extreme_data)
        assert 0.0 <= score <= 1.0, "Score should be between 0 and 1"
        assert score < 0.5, "Extreme values should have low habitability"
    
    def test_pandas_series_input(self):
        """Test that function works with pandas Series input."""
        data = pd.Series({
            'st_teff': 5780,
            'pl_rade': 1.0,
            'pl_orbper': 365.25
        })
        score = compute_habitability_index(data)
        assert 0.8 <= score <= 1.0, "Should work with pandas Series"
    
    def test_score_range(self):
        """Test that scores are always in valid range."""
        test_cases = [
            {'st_teff': 3000, 'pl_rade': 0.5, 'pl_orbper': 100},
            {'st_teff': 8000, 'pl_rade': 2.0, 'pl_orbper': 1000},
            {'st_teff': 5780, 'pl_rade': 1.5, 'pl_orbper': 500},
        ]
        
        for case in test_cases:
            score = compute_habitability_index(case)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for case {case}"
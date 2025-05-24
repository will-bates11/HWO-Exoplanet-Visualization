"""Tests for Flask application."""
import pytest
import json
import sys
import os
from unittest.mock import patch, MagicMock
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app, validate_telescope_diameter, calculate_stats


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_data():
    """Create sample exoplanet data for testing."""
    return pd.DataFrame({
        'pl_name': ['Planet A', 'Planet B', 'Planet C'],
        'st_dist': [10.0, 20.0, 30.0],
        'pl_rade': [1.0, 1.5, 2.0],
        'pl_orbper': [365, 400, 500],
        'habitability_index': [0.8, 0.6, 0.4],
        'cluster': [0, 1, 0]
    })


class TestApp:
    """Test cases for Flask application."""
    
    def test_home_page(self, client):
        """Test that home page loads."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'HWO Exoplanet Visualization' in response.data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
    
    def test_404_page(self, client):
        """Test 404 error handling."""
        response = client.get('/nonexistent')
        assert response.status_code == 404


class TestValidation:
    """Test validation functions."""
    
    def test_valid_telescope_diameter(self):
        """Test valid telescope diameter validation."""
        is_valid, diameter, error = validate_telescope_diameter("2.0")
        assert is_valid is True
        assert diameter == 2.0
        assert error == ""
    
    def test_invalid_telescope_diameter_format(self):
        """Test invalid format handling."""
        is_valid, diameter, error = validate_telescope_diameter("not_a_number")
        assert is_valid is False
        assert diameter == 0.0
        assert "Invalid telescope diameter format" in error
    
    def test_telescope_diameter_out_of_range(self):
        """Test out of range values."""
        is_valid, diameter, error = validate_telescope_diameter("200")
        assert is_valid is False
        assert "must be between" in error
    
    def test_calculate_stats(self, sample_data):
        """Test statistics calculation."""
        stats = calculate_stats(sample_data)
        assert stats['total_planets'] == 3
        assert 0.0 <= stats['avg_habitability'] <= 1.0
        assert stats['num_clusters'] == 2
    
    def test_calculate_stats_empty_data(self):
        """Test statistics with empty data."""
        empty_df = pd.DataFrame()
        stats = calculate_stats(empty_df)
        assert stats['total_planets'] == 0
        assert stats['avg_habitability'] == 0.0
        assert stats['num_clusters'] == 0


class TestFilterEndpoint:
    """Test the filter endpoint."""
    
    @patch('app.get_cached_exoplanet_data')
    @patch('app.cluster_exoplanets')
    @patch('app.create_3d_visualization')
    def test_filter_endpoint_success(self, mock_viz, mock_cluster, mock_data, client, sample_data):
        """Test successful filter request."""
        # Setup mocks
        mock_data.return_value = sample_data
        mock_cluster.return_value = sample_data
        mock_viz.return_value = {'data': [], 'layout': {}}
        
        response = client.get('/filter?telescope_diameter=2.0')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'visualization' in data
        assert 'stats' in data
    
    def test_filter_endpoint_invalid_diameter(self, client):
        """Test filter with invalid diameter."""
        response = client.get('/filter?telescope_diameter=invalid')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
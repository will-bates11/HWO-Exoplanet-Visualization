from flask import Flask, render_template, request, jsonify
from exoplanet_data import fetch_exoplanet_data
from visualization import create_3d_visualization
from clustering import cluster_exoplanets
import logging
from functools import lru_cache
import traceback
import os
import json
import re
from typing import Dict, Any, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration."""
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 12000))
    TELESCOPE_DIAMETER_MIN = 0.1
    TELESCOPE_DIAMETER_MAX = 100.0
    MAX_DISTANCE_FILTER = 50  # parsecs

# Update template directory path
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)
app.config.from_object(Config)

# Cache the exoplanet data for 1 hour
@lru_cache(maxsize=1)
def get_cached_exoplanet_data():
    """Fetch and cache exoplanet data."""
    return fetch_exoplanet_data()

@app.route('/')
def home():
    """Render the home page with initial visualization."""
    try:
        exoplanet_data = get_cached_exoplanet_data()
        exoplanet_data = cluster_exoplanets(exoplanet_data)
        visualization_html = create_3d_visualization(exoplanet_data)
        
        return render_template(
            'index.html',
            visualization=visualization_html,
            error=None
        )
    except Exception as e:
        logger.error(f"Error in home route: {traceback.format_exc()}")
        return render_template(
            'index.html',
            visualization=None,
            error="Failed to load exoplanet data. Please try again later."
        )

def calculate_stats(data) -> Dict[str, Any]:
    """Calculate summary statistics for the dataset."""
    try:
        return {
            'total_planets': len(data),
            'avg_habitability': float(data['habitability_index'].mean()) if len(data) > 0 else 0.0,
            'num_clusters': int(data['cluster'].nunique()) if 'cluster' in data.columns else 0
        }
    except Exception as e:
        logger.warning(f"Error calculating stats: {e}")
        return {'total_planets': 0, 'avg_habitability': 0.0, 'num_clusters': 0}

def validate_telescope_diameter(diameter_str: str) -> Tuple[bool, float, str]:
    """Validate telescope diameter input."""
    try:
        diameter = float(diameter_str)
        if not (Config.TELESCOPE_DIAMETER_MIN <= diameter <= Config.TELESCOPE_DIAMETER_MAX):
            return False, 0.0, f'Telescope diameter must be between {Config.TELESCOPE_DIAMETER_MIN} and {Config.TELESCOPE_DIAMETER_MAX} meters'
        return True, diameter, ""
    except (ValueError, TypeError):
        return False, 0.0, "Invalid telescope diameter format"

@app.route('/filter')
def filter_data():
    """Filter and cluster exoplanet data based on telescope diameter."""
    try:
        diameter_str = request.args.get('telescope_diameter', '2.0')
        
        # Validate input
        is_valid, diameter, error_msg = validate_telescope_diameter(diameter_str)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Get cached data and apply filters
        exoplanet_data = get_cached_exoplanet_data()
        filtered_data = exoplanet_data[exoplanet_data['st_dist'] < diameter * 10]
        
        # Perform clustering on filtered data
        filtered_data = cluster_exoplanets(filtered_data)
        
        # Create visualization - get JSON data directly
        plot_data = create_3d_visualization(filtered_data, return_json=True)
        
        # Add summary statistics
        stats = calculate_stats(filtered_data)
        
        return jsonify({
            'visualization': plot_data,
            'stats': stats
        })
        
    except ValueError as ve:
        logger.warning(f"Invalid input: {str(ve)}")
        return jsonify({'error': 'Invalid input parameters'}), 400
    except Exception as e:
        logger.error(f"Error in filter route: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'HWO Exoplanet Visualization'})

if __name__ == '__main__':
    app.run(
        debug=app.config['DEBUG'],
        host=app.config['HOST'],
        port=app.config['PORT']
    )

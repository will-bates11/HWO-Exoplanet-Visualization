from flask import Flask, render_template, request, jsonify
from exoplanet_data import fetch_exoplanet_data
from visualization import create_3d_visualization
from clustering import cluster_exoplanets
import logging
from functools import lru_cache
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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

@app.route('/filter')
def filter_data():
    """Filter and cluster exoplanet data based on telescope diameter."""
    try:
        diameter = float(request.args.get('telescope_diameter', 2.0))
        
        # Validate input
        if not (0.1 <= diameter <= 100):
            return jsonify({
                'error': 'Telescope diameter must be between 0.1 and 100 meters'
            }), 400
        
        # Get cached data and apply filters
        exoplanet_data = get_cached_exoplanet_data()
        filtered_data = exoplanet_data[exoplanet_data['st_dist'] < diameter * 10]
        
        # Perform clustering on filtered data
        filtered_data = cluster_exoplanets(filtered_data)
        
        # Create visualization
        visualization_data = create_3d_visualization(filtered_data)
        
        # Add summary statistics
        stats = {
            'total_planets': len(filtered_data),
            'avg_habitability': filtered_data['habitability_index'].mean(),
            'num_clusters': filtered_data['cluster'].nunique()
        }
        
        return jsonify({
            'visualization': visualization_data,
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

if __name__ == '__main__':
    app.run(debug=True)

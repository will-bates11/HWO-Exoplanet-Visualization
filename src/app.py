from flask import Flask, render_template, request, jsonify, g
from exoplanet_data import fetch_exoplanet_data, get_filtered_data, fetch_exoplanet_data_cached
from visualization import create_3d_visualization
from clustering import cluster_exoplanets_fast, get_cluster_summary
import logging
from functools import lru_cache, wraps
import traceback
import os
import json
import re
import time
import gzip
from typing import Dict, Any, Tuple, Optional
from threading import Lock

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
    MAX_DISTANCE_FILTER = 100  # parsecs - increased range
    CACHE_TIMEOUT = 3600  # 1 hour
    ENABLE_COMPRESSION = True

# Update template directory path
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)
app.config.from_object(Config)

# Thread-safe cache for filtered data
_filtered_cache = {}
_cache_lock = Lock()

def compress_response(f):
    """Decorator to compress JSON responses for better performance."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = f(*args, **kwargs)
        
        if (app.config.get('ENABLE_COMPRESSION', True) and 
            request.headers.get('Accept-Encoding', '').find('gzip') != -1 and
            isinstance(response, tuple) and len(response) == 2):
            
            data, status_code = response
            if isinstance(data, dict):
                json_str = json.dumps(data)
                if len(json_str) > 1000:  # Only compress larger responses
                    compressed = gzip.compress(json_str.encode('utf-8'))
                    response = app.response_class(
                        compressed, 
                        status=status_code,
                        headers={'Content-Encoding': 'gzip', 'Content-Type': 'application/json'}
                    )
                    return response
        
        return response
    return decorated_function

def timing_decorator(f):
    """Decorator to log execution time for performance monitoring."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{f.__name__} executed in {execution_time:.3f}s")
        return result
    return decorated_function

@lru_cache(maxsize=128)
def get_cached_processed_data(diameter: float, max_distance: float) -> Tuple[Any, Dict[str, Any]]:
    """Cached processing of filtered and clustered data."""
    try:
        # Use optimized filtering
        filtered_data = get_filtered_data(max_distance)
        
        if len(filtered_data) == 0:
            return None, {'total_planets': 0, 'avg_habitability': 0.0, 'num_clusters': 0}
        
        # Fast clustering
        clustered_data = cluster_exoplanets_fast(filtered_data, use_cache=True)
        
        # Calculate stats
        stats = calculate_stats(clustered_data)
        
        # Add cluster summary
        cluster_summary = get_cluster_summary(clustered_data)
        stats['cluster_summary'] = cluster_summary
        
        return clustered_data, stats
        
    except Exception as e:
        logger.error(f"Error in cached processing: {e}")
        return None, {'total_planets': 0, 'avg_habitability': 0.0, 'num_clusters': 0}

@app.before_request
def before_request():
    """Set up request context with timing."""
    g.start_time = time.time()

@app.after_request
def after_request(response):
    """Log request performance metrics."""
    if hasattr(g, 'start_time'):
        request_time = time.time() - g.start_time
        if request_time > 1.0:  # Log slow requests
            logger.warning(f"Slow request: {request.path} took {request_time:.3f}s")
    
    # Add cache headers for static content
    if request.endpoint == 'static':
        response.headers['Cache-Control'] = 'public, max-age=3600'
    
    return response

@app.route('/')
@timing_decorator
def home():
    """Render the home page with initial visualization."""
    try:
        # Use cached data with default parameters
        exoplanet_data, stats = get_cached_processed_data(2.0, 50.0)
        
        if exoplanet_data is None:
            return render_template(
                'index.html',
                visualization=None,
                error="Failed to load exoplanet data. Please try again later."
            )
        
        # Create visualization
        visualization_html = create_3d_visualization(exoplanet_data)
        
        return render_template(
            'index.html',
            visualization=visualization_html,
            stats=stats,
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
    """Calculate summary statistics for the dataset with enhanced metrics."""
    try:
        if len(data) == 0:
            return {'total_planets': 0, 'avg_habitability': 0.0, 'num_clusters': 0}
        
        # Basic stats
        stats = {
            'total_planets': len(data),
            'avg_habitability': float(data['habitability_index'].mean()),
            'num_clusters': int(data['cluster'].nunique()) if 'cluster' in data.columns else 0
        }
        
        # Enhanced stats
        if len(data) > 0:
            stats.update({
                'median_distance': float(data['st_dist'].median()),
                'max_habitability': float(data['habitability_index'].max()),
                'planets_in_habitable_zone': int((data['habitability_index'] > 0.5).sum()),
                'avg_radius': float(data['pl_rade'].mean()) if 'pl_rade' in data.columns else 0.0,
                'distance_range': {
                    'min': float(data['st_dist'].min()),
                    'max': float(data['st_dist'].max())
                }
            })
        
        return stats
        
    except Exception as e:
        logger.warning(f"Error calculating enhanced stats: {e}")
        return {'total_planets': len(data), 'avg_habitability': 0.0, 'num_clusters': 0}

def validate_telescope_diameter(diameter_str: str) -> Tuple[bool, float, str]:
    """Enhanced telescope diameter validation."""
    try:
        # Clean input
        diameter_str = diameter_str.strip()
        
        # Handle scientific notation
        diameter = float(diameter_str)
        
        if not (Config.TELESCOPE_DIAMETER_MIN <= diameter <= Config.TELESCOPE_DIAMETER_MAX):
            return False, 0.0, f'Telescope diameter must be between {Config.TELESCOPE_DIAMETER_MIN} and {Config.TELESCOPE_DIAMETER_MAX} meters'
        
        return True, diameter, ""
        
    except (ValueError, TypeError):
        return False, 0.0, "Invalid telescope diameter format. Please enter a valid number."

@app.route('/filter')
@timing_decorator
@compress_response
def filter_data():
    """Optimized filtering and clustering with enhanced caching."""
    try:
        diameter_str = request.args.get('telescope_diameter', '2.0')
        use_fast_clustering = request.args.get('fast', 'true').lower() == 'true'
        
        # Validate input
        is_valid, diameter, error_msg = validate_telescope_diameter(diameter_str)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Calculate max distance based on telescope diameter (optimized formula)
        max_distance = min(diameter * 15, Config.MAX_DISTANCE_FILTER)  # Enhanced scaling
        
        # Use cached processed data
        filtered_data, stats = get_cached_processed_data(diameter, max_distance)
        
        if filtered_data is None:
            return jsonify({'error': 'No data available for the specified parameters'}), 404
        
        # Create visualization - get JSON data directly for better performance
        try:
            plot_data = create_3d_visualization(filtered_data, return_json=True)
        except Exception as viz_error:
            logger.warning(f"Visualization error: {viz_error}")
            plot_data = {'data': [], 'layout': {}}
        
        # Prepare response
        response_data = {
            'visualization': plot_data,
            'stats': stats,
            'processing_info': {
                'telescope_diameter': diameter,
                'max_distance': max_distance,
                'fast_clustering': use_fast_clustering,
                'data_points': len(filtered_data)
            }
        }
        
        return jsonify(response_data)
        
    except ValueError as ve:
        logger.warning(f"Invalid input: {str(ve)}")
        return jsonify({'error': 'Invalid input parameters'}), 400
    except Exception as e:
        logger.error(f"Error in filter route: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error. Please try again.'}), 500

@app.route('/api/stats')
@timing_decorator
def get_stats():
    """Get dataset statistics without visualization data."""
    try:
        diameter = float(request.args.get('telescope_diameter', '2.0'))
        max_distance = min(diameter * 15, Config.MAX_DISTANCE_FILTER)
        
        # Get cached stats
        _, stats = get_cached_processed_data(diameter, max_distance)
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error in stats route: {e}")
        return jsonify({'error': 'Failed to retrieve statistics'}), 500

@app.route('/api/clusters')
@timing_decorator
def get_cluster_info():
    """Get detailed cluster information."""
    try:
        diameter = float(request.args.get('telescope_diameter', '2.0'))
        max_distance = min(diameter * 15, Config.MAX_DISTANCE_FILTER)
        
        filtered_data, _ = get_cached_processed_data(diameter, max_distance)
        
        if filtered_data is None:
            return jsonify({'clusters': {}})
        
        cluster_summary = get_cluster_summary(filtered_data)
        
        return jsonify({'clusters': cluster_summary})
        
    except Exception as e:
        logger.error(f"Error in cluster info route: {e}")
        return jsonify({'error': 'Failed to retrieve cluster information'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.route('/health')
def health_check():
    """Enhanced health check endpoint with system status."""
    try:
        # Quick data availability check
        test_data = fetch_exoplanet_data_cached(10.0)  # Small sample
        data_available = len(test_data) > 0
        
        return jsonify({
            'status': 'healthy' if data_available else 'degraded',
            'service': 'HWO Exoplanet Visualization',
            'data_available': data_available,
            'cache_entries': len(_filtered_cache),
            'version': '2.0-optimized'
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'service': 'HWO Exoplanet Visualization',
            'error': str(e)
        }), 503

@app.route('/api/cache/clear')
def clear_cache():
    """Clear application caches (for development/debugging)."""
    try:
        global _filtered_cache
        with _cache_lock:
            _filtered_cache.clear()
        
        # Clear LRU caches
        get_cached_processed_data.cache_clear()
        fetch_exoplanet_data_cached.cache_clear()
        
        return jsonify({'message': 'Caches cleared successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Failed to clear caches: {e}'}), 500

if __name__ == '__main__':
    logger.info("Starting HWO Exoplanet Visualization (Optimized)")
    app.run(
        debug=app.config['DEBUG'],
        host=app.config['HOST'],
        port=app.config['PORT'],
        threaded=True  # Enable threading for better performance
    )

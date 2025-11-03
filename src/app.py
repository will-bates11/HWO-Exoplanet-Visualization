from flask import Flask, render_template, request, jsonify, g, Response, make_response
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
import io
import csv
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from threading import Lock

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security imports
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    from flask_talisman import Talisman
    SECURITY_ENABLED = True
except ImportError:
    logger.warning("Security libraries not available. Running without rate limiting and security headers.")
    SECURITY_ENABLED = False

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

# Security enhancements
if SECURITY_ENABLED:
    # Content Security Policy and security headers
    Talisman(app, content_security_policy={
        'default-src': "'self'",
        'script-src': "'self' 'unsafe-inline'",
        'style-src': "'self' 'unsafe-inline'",
        'img-src': "'self' data:",
    })
    
    # Rate limiting
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
else:
    # Dummy limiter for when security libraries aren't available
    class DummyLimiter:
        def limit(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    limiter = DummyLimiter()

# Thread-safe cache for filtered data
_filtered_cache = {}
_cache_lock = Lock()

# Input validation functions
def validate_numeric_param(param_name, value, min_val=None, max_val=None, default=None):
    """Validate numeric parameters with optional min/max bounds."""
    try:
        num = float(value)
        if min_val is not None and num < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}")
        if max_val is not None and num > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}")
        return num
    except (ValueError, TypeError):
        if default is not None:
            return default
        raise ValueError(f"Invalid {param_name}: must be a number")

def validate_string_param(param_name, value, max_length=100, allowed_chars=None):
    """Validate string parameters with length and character restrictions."""
    if not isinstance(value, str):
        raise ValueError(f"Invalid {param_name}: must be a string")
    if len(value) > max_length:
        raise ValueError(f"{param_name} too long (max {max_length} characters)")
    if allowed_chars and not all(c in allowed_chars for c in value):
        raise ValueError(f"{param_name} contains invalid characters")
    return value

def sanitize_filters():
    """Extract and validate filter parameters from request."""
    try:
        telescope_diameter = validate_numeric_param(
            'telescope_diameter', 
            request.args.get('telescope_diameter', 6.5), 
            min_val=0.1, max_val=50, default=6.5
        )
        max_distance = validate_numeric_param(
            'max_distance', 
            request.args.get('max_distance', 100), 
            min_val=1, max_val=1000, default=100
        )
        min_habitability = validate_numeric_param(
            'min_habitability', 
            request.args.get('min_habitability', 0.1), 
            min_val=0, max_val=1, default=0.1
        )
        color_by = validate_string_param(
            'color_by', 
            request.args.get('color_by', 'habitability'), 
            max_length=20, 
            allowed_chars='abcdefghijklmnopqrstuvwxyz_'
        )
        
        # Validate color_by options
        valid_color_options = ['habitability', 'distance', 'radius', 'temperature', 'cluster']
        if color_by not in valid_color_options:
            color_by = 'habitability'
            
        return {
            'telescope_diameter': telescope_diameter,
            'max_distance': max_distance,
            'min_habitability': min_habitability,
            'color_by': color_by
        }
    except ValueError as e:
        logger.warning(f"Invalid filter parameters: {e}")
        raise

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
@limiter.limit("30 per minute")
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
@limiter.limit("10 per minute")
@timing_decorator
@compress_response
def filter_data():
    """Optimized filtering and clustering with enhanced caching."""
    try:
        diameter_str = request.args.get('telescope_diameter', '2.0')
        use_fast_clustering = request.args.get('fast', 'true').lower() == 'true'
        
        # Pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 100))
        per_page = min(per_page, 1000)  # Max 1000 items per page
        
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

@app.route('/api/planets')
@limiter.limit("10 per minute")
@timing_decorator
@compress_response
def get_planets():
    """Get paginated planet data for the current filters."""
    try:
        # Get filter parameters
        diameter_str = request.args.get('telescope_diameter', '2.0')
        max_distance = float(request.args.get('max_distance', '100'))
        min_habitability = float(request.args.get('min_habitability', '0'))
        
        # Pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        per_page = min(max(per_page, 10), 200)  # Between 10 and 200 items per page
        
        # Validate input
        is_valid, diameter, error_msg = validate_telescope_diameter(diameter_str)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Get filtered data
        filtered_data = get_filtered_data(max_distance)
        if filtered_data is None:
            return jsonify({'error': 'No data available'}), 404
        
        # Apply habitability filter
        if min_habitability > 0:
            filtered_data = filtered_data[filtered_data['habitability_index'] >= min_habitability]
        
        # Convert to list of dicts for JSON serialization
        total_count = len(filtered_data)
        
        # Calculate pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        if start_idx >= total_count:
            paginated_data = []
        else:
            paginated_data = filtered_data.iloc[start_idx:end_idx].to_dict('records')
        
        # Calculate pagination metadata
        total_pages = (total_count + per_page - 1) // per_page  # Ceiling division
        
        response_data = {
            'planets': paginated_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_count': total_count,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            },
            'filters': {
                'telescope_diameter': diameter,
                'max_distance': max_distance,
                'min_habitability': min_habitability
            }
        }
        
        return jsonify(response_data)
        
    except ValueError as ve:
        logger.warning(f"Invalid input: {str(ve)}")
        return jsonify({'error': 'Invalid input parameters'}), 400
    except Exception as e:
        logger.error(f"Error in planets route: {traceback.format_exc()}")
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
@limiter.limit("60 per minute")
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

@app.route('/api/export/csv')
@limiter.limit("5 per minute")
@timing_decorator
def export_csv():
    """Export filtered exoplanet data as CSV."""
    try:
        # Get filter parameters
        diameter_str = request.args.get('telescope_diameter', '2.0')
        max_distance = float(request.args.get('max_distance', '100'))
        min_habitability = float(request.args.get('min_habitability', '0'))
        limit = int(request.args.get('limit', 10000))  # Default 10k rows max for exports
        limit = min(limit, 50000)  # Hard limit of 50k rows
        
        # Validate input
        is_valid, diameter, error_msg = validate_telescope_diameter(diameter_str)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Get filtered data
        filtered_data = get_filtered_data(max_distance)
        if filtered_data is None or len(filtered_data) == 0:
            return jsonify({'error': 'No data available for the specified parameters'}), 404
        
        # Apply additional filters
        if min_habitability > 0:
            filtered_data = filtered_data[filtered_data['habitability_index'] >= min_habitability]
        
        # Apply limit for performance
        if len(filtered_data) > limit:
            filtered_data = filtered_data.head(limit)
        
        # Select columns for export (exclude internal fields)
        export_columns = [
            'pl_name', 'st_dist', 'pl_rade', 'st_teff', 'pl_orbper', 
            'habitability_index', 'pl_bmassj', 'st_mass', 'pl_eqt', 
            'pl_insol', 'pl_orbeccen', 'st_age'
        ]
        available_columns = [col for col in export_columns if col in filtered_data.columns]
        export_data = filtered_data[available_columns].copy()
        
        # Create CSV response
        output = io.StringIO()
        export_data.to_csv(output, index=False)
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=exoplanets_d{diameter}_dist{max_distance}.csv'
        
        return response
        
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        return jsonify({'error': 'Failed to export data'}), 500

@app.route('/api/export/json')
@limiter.limit("5 per minute")
@timing_decorator
def export_json():
    """Export filtered exoplanet data as JSON."""
    try:
        # Get filter parameters
        diameter_str = request.args.get('telescope_diameter', '2.0')
        max_distance = float(request.args.get('max_distance', '100'))
        min_habitability = float(request.args.get('min_habitability', '0'))
        limit = int(request.args.get('limit', 10000))  # Default 10k rows max for exports
        limit = min(limit, 50000)  # Hard limit of 50k rows
        
        # Validate input
        is_valid, diameter, error_msg = validate_telescope_diameter(diameter_str)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Get filtered data
        filtered_data = get_filtered_data(max_distance)
        if filtered_data is None or len(filtered_data) == 0:
            return jsonify({'error': 'No data available for the specified parameters'}), 404
        
        # Apply additional filters
        if min_habitability > 0:
            filtered_data = filtered_data[filtered_data['habitability_index'] >= min_habitability]
        
        # Apply limit for performance
        if len(filtered_data) > limit:
            filtered_data = filtered_data.head(limit)
        
        # Select columns for export
        export_columns = [
            'pl_name', 'st_dist', 'pl_rade', 'st_teff', 'pl_orbper', 
            'habitability_index', 'pl_bmassj', 'st_mass', 'pl_eqt', 
            'pl_insol', 'pl_orbeccen', 'st_age'
        ]
        available_columns = [col for col in export_columns if col in filtered_data.columns]
        export_data = filtered_data[available_columns].copy()
        
        # Convert to JSON-serializable format
        json_data = {
            'metadata': {
                'telescope_diameter': diameter,
                'max_distance': max_distance,
                'min_habitability': min_habitability,
                'total_planets': len(export_data),
                'export_timestamp': time.time()
            },
            'data': export_data.to_dict('records')
        }
        
        response = make_response(json.dumps(json_data, indent=2))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = f'attachment; filename=exoplanets_d{diameter}_dist{max_distance}.json'
        
        return response
        
    except Exception as e:
        logger.error(f"Error exporting JSON: {e}")
        return jsonify({'error': 'Failed to export data'}), 500

@app.route('/api/search')
@limiter.limit("20 per minute")
@timing_decorator
def search_planets():
    """Search for planets by name with autocomplete."""
    try:
        query = request.args.get('q', '').strip().lower()
        limit = int(request.args.get('limit', 10))
        
        if len(query) < 2:
            return jsonify({'results': []})
        
        # Get base data
        data = fetch_exoplanet_data_cached()
        if data is None:
            return jsonify({'results': []})
        
        # Filter by name (case-insensitive partial match)
        matches = data[data['pl_name'].str.lower().str.contains(query, na=False)]
        
        # Sort by relevance (exact matches first, then starts with, then contains)
        def relevance_score(name):
            name_lower = name.lower()
            if name_lower == query:
                return 0
            elif name_lower.startswith(query):
                return 1
            else:
                return 2
        
        matches['relevance'] = matches['pl_name'].apply(relevance_score)
        matches = matches.sort_values(['relevance', 'pl_name']).head(limit)
        
        # Format results
        results = []
        for _, planet in matches.iterrows():
            results.append({
                'name': planet['pl_name'],
                'distance': float(planet['st_dist']) if pd.notna(planet['st_dist']) else None,
                'radius': float(planet['pl_rade']) if pd.notna(planet['pl_rade']) else None,
                'habitability': float(planet['habitability_index']) if pd.notna(planet['habitability_index']) else 0.0
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error searching planets: {e}")
        return jsonify({'error': 'Search failed'}), 500

if __name__ == '__main__':
    logger.info("Starting HWO Exoplanet Visualization (Optimized)")
    app.run(
        debug=app.config['DEBUG'],
        host=app.config['HOST'],
        port=app.config['PORT'],
        threaded=True  # Enable threading for better performance
    )

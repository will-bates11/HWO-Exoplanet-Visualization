import plotly.graph_objs as go
import pandas as pd
import numpy as np
import logging
from typing import Union, Dict, Any, Optional, Tuple
from functools import lru_cache
import json

logger = logging.getLogger(__name__)

# Visualization configuration constants
VIZ_CONFIG = {
    'max_points': 2000,  # Limit points for performance
    'min_marker_size': 3,
    'max_marker_size': 25,
    'default_opacity': 0.8,
    'cluster_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'habitability_colorscale': 'Viridis'
}

@lru_cache(maxsize=64)
def generate_cluster_colors(num_clusters: int) -> list:
    """Generate colors for clusters with caching."""
    if num_clusters <= len(VIZ_CONFIG['cluster_colors']):
        return VIZ_CONFIG['cluster_colors'][:num_clusters]
    
    # Generate additional colors if needed
    colors = VIZ_CONFIG['cluster_colors'].copy()
    import colorsys
    
    for i in range(len(colors), num_clusters):
        hue = (i * 0.618033988749895) % 1  # Golden ratio for good color distribution
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        colors.append(f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})')
    
    return colors

def optimize_data_for_visualization(data: pd.DataFrame) -> pd.DataFrame:
    """Optimize dataset for visualization performance."""
    if len(data) <= VIZ_CONFIG['max_points']:
        return data
    
    logger.info(f"Optimizing {len(data)} points for visualization (target: {VIZ_CONFIG['max_points']})")
    
    try:
        # Stratified sampling to maintain cluster representation
        if 'cluster' in data.columns and data['cluster'].nunique() > 1:
            # Sample proportionally from each cluster
            samples_per_cluster = max(1, VIZ_CONFIG['max_points'] // data['cluster'].nunique())
            
            sampled_data = []
            for cluster in data['cluster'].unique():
                cluster_data = data[data['cluster'] == cluster]
                if len(cluster_data) > samples_per_cluster:
                    # Weighted sampling - prefer high habitability and close distance
                    weights = cluster_data['habitability_index'] * (1 / (cluster_data['st_dist'] + 0.1))
                    sample = cluster_data.sample(n=min(samples_per_cluster, len(cluster_data)), 
                                               weights=weights, random_state=42)
                else:
                    sample = cluster_data
                sampled_data.append(sample)
            
            optimized_data = pd.concat(sampled_data, ignore_index=True)
        else:
            # Simple weighted random sampling
            weights = data['habitability_index'] * (1 / (data['st_dist'] + 0.1))
            optimized_data = data.sample(n=VIZ_CONFIG['max_points'], weights=weights, random_state=42)
        
        logger.info(f"Optimized dataset to {len(optimized_data)} points")
        return optimized_data
        
    except Exception as e:
        logger.warning(f"Error in data optimization: {e}, using random sample")
        return data.sample(n=min(VIZ_CONFIG['max_points'], len(data)), random_state=42)

def create_hover_text_vectorized(data: pd.DataFrame) -> list:
    """Vectorized hover text creation for better performance."""
    hover_texts = []
    
    # Pre-format columns to avoid repeated formatting
    names = data['pl_name'].fillna('Unknown')
    distances = data['st_dist'].fillna(0).round(2)
    radii = data['pl_rade'].fillna(0).round(2)
    periods = data['pl_orbper'].fillna(0).round(1)
    habitability = data['habitability_index'].fillna(0).round(3)
    
    # Include cluster information if available
    has_clusters = 'cluster' in data.columns
    if has_clusters:
        clusters = data['cluster'].fillna(-1)
    
    # Vectorized string operations
    for i in range(len(data)):
        text_parts = [
            f"<b>{names.iloc[i]}</b>",
            f"Distance: {distances.iloc[i]} pc",
            f"Radius: {radii.iloc[i]} R⊕",
            f"Period: {periods.iloc[i]} days",
            f"Habitability: {habitability.iloc[i]}"
        ]
        
        if has_clusters and clusters.iloc[i] != -1:
            text_parts.append(f"Cluster: {int(clusters.iloc[i])}")
        
        hover_texts.append("<br>".join(text_parts))
    
    return hover_texts

def calculate_marker_sizes(data: pd.DataFrame) -> np.ndarray:
    """Calculate optimized marker sizes based on planet characteristics."""
    try:
        # Primary: use planet radius if available
        if 'pl_rade' in data.columns and not data['pl_rade'].isna().all():
            size_base = data['pl_rade'].fillna(1.0)
        # Fallback: use mass if available
        elif 'pl_bmassj' in data.columns and not data['pl_bmassj'].isna().all():
            size_base = data['pl_bmassj'].fillna(1.0) 
        else:
            # Default uniform size
            return np.full(len(data), 8)
        
        # Apply log scaling for better visual distribution
        size_base = np.log1p(size_base)  # log1p handles zeros gracefully
        
        # Normalize to desired range
        min_size, max_size = VIZ_CONFIG['min_marker_size'], VIZ_CONFIG['max_marker_size']
        
        if size_base.max() > size_base.min():
            sizes = min_size + (size_base - size_base.min()) / (size_base.max() - size_base.min()) * (max_size - min_size)
        else:
            sizes = np.full(len(data), (min_size + max_size) / 2)
        
        return sizes
        
    except Exception as e:
        logger.warning(f"Error calculating marker sizes: {e}")
        return np.full(len(data), 8)

def create_clustered_visualization(data: pd.DataFrame) -> list:
    """Create visualization with cluster-aware coloring and enhanced performance."""
    try:
        traces = []
        
        # Get unique clusters
        clusters = data['cluster'].unique()
        cluster_colors = generate_cluster_colors(len(clusters))
        
        # Create separate trace for each cluster for better visual distinction
        for i, cluster in enumerate(clusters):
            if cluster == -1:  # Skip unassigned points
                continue
                
            cluster_data = data[data['cluster'] == cluster]
            if len(cluster_data) == 0:
                continue
            
            # Prepare coordinates
            x = cluster_data['st_dist'].fillna(0)
            y = cluster_data['pl_rade'].fillna(1)
            z = cluster_data['pl_orbper'].fillna(365)
            
            # Calculate marker properties
            sizes = calculate_marker_sizes(cluster_data)
            colors = cluster_data['habitability_index'].fillna(0)
            
            # Create hover text
            hover_text = create_hover_text_vectorized(cluster_data)
            
            trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale=VIZ_CONFIG['habitability_colorscale'],
                    opacity=VIZ_CONFIG['default_opacity'],
                    colorbar=dict(
                        title="Habitability Index",
                        titleside="right"
                    ) if i == 0 else None,  # Only show colorbar for first trace
                    line=dict(width=1, color=cluster_colors[i % len(cluster_colors)]),
                    cmin=0, cmax=1  # Fix colorbar range
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name=f'Cluster {cluster}',
                showlegend=True
            )
            traces.append(trace)
        
        # Handle unassigned points separately
        unassigned_data = data[data['cluster'] == -1]
        if len(unassigned_data) > 0:
            x = unassigned_data['st_dist'].fillna(0)
            y = unassigned_data['pl_rade'].fillna(1)
            z = unassigned_data['pl_orbper'].fillna(365)
            
            sizes = calculate_marker_sizes(unassigned_data)
            colors = unassigned_data['habitability_index'].fillna(0)
            hover_text = create_hover_text_vectorized(unassigned_data)
            
            trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale=VIZ_CONFIG['habitability_colorscale'],
                    opacity=0.5,  # Lower opacity for unassigned
                    line=dict(width=0.5, color='lightgray')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Unassigned',
                showlegend=True
            )
            traces.append(trace)
        
        return traces
        
    except Exception as e:
        logger.error(f"Error creating clustered visualization: {e}")
        return []

def create_3d_visualization(exoplanet_data: pd.DataFrame, return_json: bool = False) -> Union[str, Dict[str, Any]]:
    """
    Create an optimized 3D visualization of exoplanet data.
    
    Parameters:
    -----------
    exoplanet_data : pd.DataFrame
        DataFrame containing exoplanet data
    return_json : bool
        If True, return JSON data instead of HTML
        
    Returns:
    --------
    Union[str, Dict[str, Any]]
        HTML string or JSON data for the plot
    """
    try:
        if exoplanet_data.empty:
            logger.warning("No data provided for visualization")
            return _create_empty_plot(return_json)
        
        # Optimize data for visualization performance
        viz_data = optimize_data_for_visualization(exoplanet_data)
        
        # Create clustered or simple visualization based on data
        if 'cluster' in viz_data.columns and viz_data['cluster'].nunique() > 1:
            traces = create_clustered_visualization(viz_data)
        else:
            # Simple single-trace visualization
            traces = [_create_simple_trace(viz_data)]
        
        if not traces:
            return _create_empty_plot(return_json)
        
        # Enhanced layout
        layout = go.Layout(
            title={
                'text': f'Exoplanets Observable by HWO ({len(viz_data)} points)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            scene=dict(
                xaxis=dict(
                    title='Distance (parsecs)', 
                    backgroundcolor="rgb(240, 240, 245)",
                    gridcolor="white",
                    showspikes=False
                ),
                yaxis=dict(
                    title='Radius (Earth Radii)', 
                    backgroundcolor="rgb(240, 240, 245)",
                    gridcolor="white",
                    showspikes=False
                ),
                zaxis=dict(
                    title='Orbital Period (days)', 
                    backgroundcolor="rgb(240, 240, 245)",
                    gridcolor="white",
                    showspikes=False
                ),
                bgcolor="rgb(248, 248, 252)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)  # Better default viewing angle
                )
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            font=dict(size=12),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            # Improved performance settings
            hovermode='closest',
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(traces)}],
                            label="Show All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [i == 0 for i in range(len(traces))]}],
                            label="Show First Cluster",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ] if len(traces) > 1 else []
        )
        
        fig = go.Figure(data=traces, layout=layout)
        
        # Configure for better performance
        fig.update_layout(
            autosize=True,
            height=600
        )
        
        if return_json:
            # Return optimized JSON
            fig_dict = fig.to_dict()
            # Remove unnecessary data for smaller payload
            if 'layout' in fig_dict and 'updatemenus' in fig_dict['layout']:
                if not fig_dict['layout']['updatemenus']:
                    del fig_dict['layout']['updatemenus']
            return fig_dict
        else:
            # Configure HTML output
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'responsive': True
            }
            return fig.to_html(include_plotlyjs=True, div_id="visualization", config=config)
            
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return _create_empty_plot(return_json)

def _create_simple_trace(data: pd.DataFrame) -> go.Scatter3d:
    """Create a simple single-trace visualization."""
    x = data['st_dist'].fillna(0)
    y = data['pl_rade'].fillna(1)
    z = data['pl_orbper'].fillna(365)
    
    sizes = calculate_marker_sizes(data)
    colors = data['habitability_index'].fillna(0)
    hover_text = create_hover_text_vectorized(data)
    
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            colorscale=VIZ_CONFIG['habitability_colorscale'],
            opacity=VIZ_CONFIG['default_opacity'],
            colorbar=dict(
                title="Habitability Index",
                titleside="right"
            ),
            line=dict(width=0.5, color='DarkSlateGrey'),
            cmin=0, cmax=1
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Exoplanets',
        showlegend=False
    )

def _create_empty_plot(return_json: bool = False) -> Union[str, Dict[str, Any]]:
    """Create an optimized empty plot for error cases."""
    fig = go.Figure()
    fig.update_layout(
        title="No data available for visualization",
        annotations=[
            dict(
                text="No exoplanet data to display<br>Try adjusting your search parameters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, 
                font=dict(size=16, color="gray")
            )
        ],
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )
    
    if return_json:
        return fig.to_dict()
    else:
        return fig.to_html(include_plotlyjs=True, div_id="visualization")

@lru_cache(maxsize=32)
def get_visualization_stats(data_hash: str, num_points: int, num_clusters: int) -> Dict[str, Any]:
    """Get cached visualization statistics."""
    return {
        'points_displayed': num_points,
        'clusters': num_clusters,
        'data_hash': data_hash[:8]
    }

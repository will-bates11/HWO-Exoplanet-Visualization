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
    'max_points': 2000,
    'min_marker_size': 5,
    'max_marker_size': 30,
    'default_opacity': 0.85,
    'cluster_colors': [
        'rgb(0, 180, 255)',  # Bright blue
        'rgb(0, 255, 157)',  # Cyan
        'rgb(255, 187, 0)',  # Gold
        'rgb(255, 68, 68)',  # Red
        'rgb(170, 0, 255)',  # Purple
        'rgb(255, 170, 0)',  # Orange
    ],
    'habitability_colorscale': [
        [0, "rgb(40, 0, 80)"],
        [0.2, "rgb(70, 0, 120)"],
        [0.4, "rgb(90, 30, 150)"],
        [0.6, "rgb(110, 50, 180)"],
        [0.8, "rgb(130, 90, 210)"],
        [1, "rgb(150, 130, 240)"]
    ]
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
                        title="Habitability Index"
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
    Create an optimized 3D visualization of exoplanet data with a space-themed design.
    
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
            traces = [_create_simple_trace(viz_data)]
        
        if not traces:
            return _create_empty_plot(return_json)
        
        # Enhanced layout with space theme
        layout = go.Layout(
            title={
                'text': f'Exoplanets Observable by HWO ({len(viz_data)} points)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': 'white', 'family': 'Segoe UI'}
            },
            scene=dict(
                xaxis=dict(
                    title=dict(
                        text='Distance (parsecs)',
                        font=dict(color="#3498db", size=12)
                    ),
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="rgba(52, 152, 219, 0.1)",
                    showspikes=False,
                    showgrid=True,
                    zeroline=False,
                    showline=True,
                    linecolor='rgba(52, 152, 219, 0.5)'
                ),
                yaxis=dict(
                    title=dict(
                        text='Radius (Earth Radii)',
                        font=dict(color="#3498db", size=12)
                    ),
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="rgba(52, 152, 219, 0.1)",
                    showspikes=False,
                    showgrid=True,
                    zeroline=False,
                    showline=True,
                    linecolor='rgba(52, 152, 219, 0.5)'
                ),
                zaxis=dict(
                    title=dict(
                        text='Orbital Period (days)',
                        font=dict(color="#3498db", size=12)
                    ),
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="rgba(52, 152, 219, 0.1)",
                    showspikes=False,
                    showgrid=True,
                    zeroline=False,
                    showline=True,
                    linecolor='rgba(52, 152, 219, 0.5)'
                ),
                bgcolor="rgb(0, 0, 0)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            font=dict(size=12, color="white", family="Segoe UI"),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(0, 0, 0, 0.8)",
                bordercolor="rgba(52, 152, 219, 0.3)",
                borderwidth=1,
                font=dict(color="white", size=10)
            ),
            hovermode='closest',
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
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
    """Create a simple single-trace visualization with enhanced marker styling."""
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
            colorscale=[
                [0, "rgb(40, 0, 80)"],
                [0.2, "rgb(70, 0, 120)"],
                [0.4, "rgb(90, 30, 150)"],
                [0.6, "rgb(110, 50, 180)"],
                [0.8, "rgb(130, 90, 210)"],
                [1, "rgb(150, 130, 240)"]
            ],
            opacity=0.8,
            colorbar=dict(
                title=dict(
                    text="Habitability Index",
                    font=dict(color="white", size=12)
                ),
                tickfont=dict(color="white"),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(52, 152, 219, 0.3)"
            ),
            line=dict(width=1, color='rgba(52, 152, 219, 0.5)'),
            cmin=0, 
            cmax=1
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

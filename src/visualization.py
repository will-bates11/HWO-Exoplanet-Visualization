import plotly.graph_objs as go
import pandas as pd
import numpy as np
import logging
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)

def create_3d_visualization(exoplanet_data: pd.DataFrame, return_json: bool = False) -> Union[str, Dict[str, Any]]:
    """
    Create a 3D visualization of exoplanet data.
    
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
        
        # Prepare data with proper handling of missing values
        x = exoplanet_data['st_dist'].fillna(0)
        y = exoplanet_data['pl_rade'].fillna(1)  # Default to Earth radius
        z = exoplanet_data['pl_orbper'].fillna(365)  # Default to Earth orbital period
        
        # Handle size data - use radius if mass is not available
        size_data = exoplanet_data['pl_bmassj'].fillna(exoplanet_data['pl_rade'].fillna(1))
        # Scale size for better visualization (clamp between 3 and 20)
        size = np.clip(size_data * 5 + 3, 3, 20)
        
        color = exoplanet_data['habitability_index'].fillna(0)
        
        # Create hover text with planet information
        hover_text = []
        for idx, row in exoplanet_data.iterrows():
            text = f"<b>{row.get('pl_name', 'Unknown')}</b><br>"
            text += f"Distance: {row.get('st_dist', 'N/A'):.2f} pc<br>"
            text += f"Radius: {row.get('pl_rade', 'N/A'):.2f} R⊕<br>"
            text += f"Period: {row.get('pl_orbper', 'N/A'):.1f} days<br>"
            text += f"Habitability: {row.get('habitability_index', 0):.3f}<br>"
            if 'cluster' in row and row['cluster'] != -1:
                text += f"Cluster: {row['cluster']}"
            hover_text.append(text)
        
        trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title="Habitability Index",
                    titleside="right"
                ),
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Exoplanets'
        )
        
        layout = go.Layout(
            title={
                'text': 'Exoplanets Observable by HWO',
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis=dict(title='Distance (parsecs)', backgroundcolor="rgb(230, 230,230)"),
                yaxis=dict(title='Radius (Earth Radii)', backgroundcolor="rgb(230, 230,230)"),
                zaxis=dict(title='Orbital Period (days)', backgroundcolor="rgb(230, 230,230)"),
                bgcolor="rgb(244, 244, 248)"
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            font=dict(size=12)
        )
        
        fig = go.Figure(data=[trace], layout=layout)
        
        if return_json:
            return fig.to_dict()
        else:
            return fig.to_html(include_plotlyjs=True, div_id="visualization")
            
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return _create_empty_plot(return_json)

def _create_empty_plot(return_json: bool = False) -> Union[str, Dict[str, Any]]:
    """Create an empty plot for error cases."""
    fig = go.Figure()
    fig.update_layout(
        title="No data available",
        annotations=[
            dict(
                text="No exoplanet data to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
        ]
    )
    
    if return_json:
        return fig.to_dict()
    else:
        return fig.to_html(include_plotlyjs=True, div_id="visualization")

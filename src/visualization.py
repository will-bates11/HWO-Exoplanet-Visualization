import plotly.graph_objs as go

def create_3d_visualization(exoplanet_data):
    x = exoplanet_data['st_dist']
    y = exoplanet_data['pl_rade']
    z = exoplanet_data['pl_orbper']
    size = exoplanet_data['pl_bmassj']
    color = exoplanet_data['habitability_index']
    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            colorscale='Viridis',
            opacity=0.8
        )
    )
    layout = go.Layout(
        title='Exoplanets Observable by HWO',
        scene=dict(
            xaxis_title='Distance (parsecs)',
            yaxis_title='Radius (Earth Radii)',
            zaxis_title='Orbital Period (days)'
        )
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig.to_html()

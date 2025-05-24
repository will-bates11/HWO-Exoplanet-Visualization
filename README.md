# HWO Exoplanet Visualization

## Author: William Bates

A web application for visualizing exoplanets observable by the Habitable Worlds Observatory (HWO) using data from NASA's Exoplanet Archive.

## Features

- **3D Interactive Visualization**: Explore exoplanets in a 3D space showing distance, radius, and orbital period
- **Habitability Index**: Custom scoring system based on Earth-like characteristics
- **Machine Learning Clustering**: Automatic grouping of similar exoplanets
- **Real-time Filtering**: Adjust telescope diameter to see observable planets
- **Responsive Design**: Works on desktop and mobile devices
- **Data Caching**: Efficient data management with automatic cache refresh
- **Enhanced Error Handling**: Robust error handling and user feedback
- **Type Safety**: Full type hints throughout the codebase

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/will-bates11/HWO-Exoplanet-Visualization.git
cd HWO-Exoplanet-Visualization
```

2. Set up a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

5. Run the application:
```bash
cd src
python app.py
```

6. Open your browser and navigate to `http://localhost:12000`

## Usage

### Web Interface

1. **Telescope Diameter Slider**: Adjust the telescope diameter (1-10 meters) to filter observable exoplanets
2. **3D Visualization**: 
   - Rotate: Click and drag
   - Zoom: Mouse wheel or pinch
   - Pan: Shift + click and drag
3. **Statistics Panel**: View real-time statistics about filtered data
4. **Hover Information**: Hover over planets to see detailed information

### API Endpoints

- `GET /`: Main application page
- `GET /filter?telescope_diameter=X`: Get filtered exoplanet data
- `GET /health`: Health check endpoint

## Architecture

### Backend Components

- **Flask Application** (`app.py`): Main web server and API endpoints
- **Data Fetcher** (`exoplanet_data.py`): NASA API integration with caching
- **Visualization** (`visualization.py`): Plotly-based 3D plotting
- **Clustering** (`clustering.py`): K-means clustering with automatic optimization
- **Utils** (`utils.py`): Habitability index calculation

### Frontend Components

- **HTML Template** (`templates/index.html`): Main user interface
- **CSS Styling** (`static/style.css`): Responsive design with animations
- **JavaScript**: Real-time updates and error handling

## Data Sources

- **NASA Exoplanet Archive**: Primary data source for exoplanet characteristics
- **Cached Data**: Local caching for improved performance (refreshed daily)

## Habitability Index

The habitability index is calculated based on three factors:

1. **Stellar Temperature** (50% weight): How similar the host star is to our Sun
2. **Planet Radius** (30% weight): How Earth-like the planet size is
3. **Orbital Period** (20% weight): How similar the orbital period is to Earth's

Score ranges from 0 (least habitable) to 1 (most Earth-like).

## Configuration

Environment variables (see `.env.example`):

- `FLASK_DEBUG`: Enable/disable debug mode
- `FLASK_HOST`: Server host (default: 0.0.0.0)
- `FLASK_PORT`: Server port (default: 12000)
- `TELESCOPE_DIAMETER_MIN/MAX`: Valid telescope diameter range
- `CACHE_DURATION_DAYS`: How long to cache NASA API data

## Testing

Run the test suite:

```bash
# Install test dependencies (included in requirements.txt)
pip install pytest

# Run tests
pytest tests/
```

## Development

### Code Quality

- Type hints throughout the codebase
- Comprehensive error handling
- Logging for debugging and monitoring
- Input validation and sanitization

### Performance Optimizations

- Data caching with automatic refresh
- Efficient clustering algorithms
- Responsive frontend with loading states
- Optimized API responses

## Troubleshooting

### Common Issues

1. **NASA API Timeout**: The app uses cached data as fallback
2. **Large Dataset Performance**: Clustering is optimized for datasets up to 10,000 planets
3. **Browser Compatibility**: Requires modern browser with JavaScript enabled

### Logs

Check the console output for detailed error messages and debugging information.

## License

MIT License

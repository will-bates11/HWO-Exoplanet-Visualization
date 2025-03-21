# Habitable Worlds Observatory (HWO) Exoplanet Visualization with Enhancements

## Author: William Bates

### Overview

This project implements an interactive 3D visualization tool to map known exoplanets and explore their potential observability with NASA's **Habitable Worlds Observatory (HWO)**. It features dynamic filters for telescope parameters and calculates a habitability index based on exoplanet characteristics.

### Key Features
- **3D Exoplanet Visualization**: Allows users to interact with exoplanet data, visualizing their distance, radius, and orbital period.
- **Dynamic Filtering**: Adjust the HWO telescope diameter and instantly see how it impacts the observability of exoplanets.
- **Habitability Index**: Calculate and visualize the habitability index for each exoplanet.
- **Clustering**: Exoplanets are clustered into different groups based on their physical characteristics.

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/will-bates11/HWO-Exoplanet-Visualization.git
cd HWO-Exoplanet-Visualization
```

#### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### Running the App

To start the 3D visualization app, run:

```bash
python src/app.py
```

### License

MIT License

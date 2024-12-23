# HERE-roadreopening

## Overview
This Python script processes probe data to determine whether a closed road has been reopened based on the trajectories of vehicles. It utilizes geographic data handling libraries to compute distances and headings between points and road segments, and identifies valid trajectories based on specified criteria.

[ArcGIS Visualization Netherlands](https://here.maps.arcgis.com/home/item.html?id=49e4c818a7234523bc37dfb5677ce68a)

[ArcGIS Visualization Poland](https://here.maps.arcgis.com/home/item.html?id=625bffe4175145418ad12f55ad1d2760)
## Features
- **Distance Calculation**: Determines if probe points are near a closed road using a specified distance threshold.
- **Heading Comparison**: Calculates the heading of the road and compares it with the heading of the probe points to ensure consistency.
- **Session Grouping**: Groups probe data by session ID to analyze trajectories over time.
- **Data Export**: Saves the resulting trajectories as a GeoJSON file for further analysis or visualization.

## Requirements
- Python 3.x
- Required libraries:
  - `geopandas`
  - `pandas`
  - `shapely`
  - `tqdm`
  - `logging`
  
You can install the necessary libraries using pip:

```bash
pip install geopandas pandas shapely tqdm
```

## Configuration
Before running the script, you can adjust the following constants in the code:
- `DISTANCE_THRESHOLD`: Maximum distance in meters for a point to be considered near the road (default: 10.0 meters).
- `DELTA_HEADING_MAX`: Maximum allowable heading difference in degrees (default: 30.0 degrees).
- `MIN_TRAJECTORIES`: Minimum number of trajectories near a closed road to be considered reopened (default: 3).

## Data Input
The script requires two input files:
1. **Closed Road GeoJSON**: Path to the GeoJSON file representing the closed road (e.g., `line_string.geojson`).
2. **Probe Data CSV**: Path to the CSV file containing probe data with `latitude` and `longitude` columns, as well as a `session_id` column for grouping.

## Running the Script
To execute the script, run:

```bash
python main.py
```

## Output
The script generates a GeoJSON file named `trajectories_by_session.geojson` containing the valid trajectories grouped by session ID. It logs the number of trajectories processed and indicates whether the road is considered open or closed based on the analysis.

## Logging
The script logs key information and warnings during execution, providing insights into the processing steps and any potential issues.

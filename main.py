import geopandas as gpd
import pandas as pd
import math
from shapely.geometry import Point, LineString
from tqdm import tqdm
import logging

# Configurations and Info
# EPSG:4326 (WGS 84) uses degrees of latitude and longitude, suitable for global mapping but not for accurate distance calculations. 
# EPSG:3857 is a projected coordinate system that uses meters, allowing for precise distance measurements.

DISTANCE_THRESHOLD = 10.0   # Max distance in meters for a point to be considered near the road
DELTA_HEADING_MAX = 30.0    # Max allowable heading difference in degrees
MIN_TRAJECTORIES = 3    # Min number of trajectories near a closed road to be considered reopened 
MIN_SPEED = 2 # Min speed in kph of vehicle to be considered a data point

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper Functions
def distance_to_line(line, point):
    """
    Calculate distance from a point to a line in meters.

    Args:
    - line: Shapely LineString object representing the road.
    - point: Shapely Point object representing the probe point.

    Returns:
    - Distance in meters from the point to the line.
    """
    return line.distance(point)

def compute_road_heading(segment, point):
    """
    Compute heading of the road at a given point.

    Args:
    - segment: Shapely LineString representing the road segment.
    - point: Shapely Point object representing the probe point.

    Returns:
    - Heading in degrees, or None if calculation fails.
    """
    if not isinstance(segment, LineString) or segment.is_empty:
        return None
    try:
        nearest_point = segment.interpolate(segment.project(point))  # Find the nearest point on the line
        coords = list(segment.coords)
        for i in range(len(coords) - 1):
            start, end = Point(coords[i]), Point(coords[i + 1])
            if start.equals(nearest_point) or end.equals(nearest_point):
                heading = math.degrees(math.atan2(end.y - start.y, end.x - start.x))  # Calculate heading
                return (heading + 360) % 360  # Normalize heading to [0, 360)
    except Exception as e:
        logging.error(f"Error in compute_road_heading: {e}")
    return None

def heading_difference(h1, h2):
    """
    Compute the absolute difference between two headings.

    Args:
    - h1: Heading in degrees.
    - h2: Heading in degrees.

    Returns:
    - Absolute difference in degrees, minimized to [0, 180].
    """
    diff = abs(h1 - h2)
    return min(diff, 360 - diff)

def load_data(closed_road_path, probe_data_path):
    """
    Load closed road and probe data from specified paths.

    Args:
    - closed_road_path: Path to the closed road GeoJSON file.
    - probe_data_path: Path to the probe data CSV file.

    Returns:
    - Tuple containing closed road GeoDataFrame and probe data GeoDataFrame.
    """
    closed_road_gdf = gpd.read_file(closed_road_path).to_crs(epsg=4326)
    probe_data_df = pd.read_csv(probe_data_path)
    filtered_probe_data_df = probe_data_df[probe_data_df['speed'] >= MIN_SPEED]
    probe_data_gdf = gpd.GeoDataFrame(
        filtered_probe_data_df,
        geometry=gpd.points_from_xy(filtered_probe_data_df.longitude, filtered_probe_data_df.latitude),
        crs="EPSG:4326",
    )
    return closed_road_gdf, probe_data_gdf

def process_trajectories(closed_road_gdf, probe_data_gdf):
    """
    Process trajectories to determine valid probe points near the closed road.

    Args:
    - closed_road_gdf: GeoDataFrame for closed road.
    - probe_data_gdf: GeoDataFrame for probe data.

    Returns:
    - Dictionary of valid trajectories by session ID.
    """
    closed_road_gdf = closed_road_gdf.to_crs(epsg=3857)
    probe_data_gdf = probe_data_gdf.to_crs(epsg=3857)

    coordinates = [list(geom.coords)[0] for geom in closed_road_gdf.geometry]
    closed_road_line = LineString(coordinates)

    trajectories_by_session = {}
    for session_id, group in tqdm(probe_data_gdf.groupby("session_id"), desc="Processing sessions"):
        valid_probes = [] 

        for point in group.geometry:
            dist = distance_to_line(closed_road_line, point)
            if dist <= DISTANCE_THRESHOLD: 
                segment_heading = compute_road_heading(closed_road_line, point)
                road_heading = compute_road_heading(closed_road_line, point)
                if segment_heading is not None and road_heading is not None:
                    if heading_difference(road_heading, segment_heading) <= DELTA_HEADING_MAX:
                        valid_probes.append(point) 

        # Debugging: Check valid probes
        # logging.info(f"Session {session_id}: valid_probes = {len(valid_probes)}")

        # Build trajectory for this session if there are enough valid points
        if len(valid_probes) > 1: 
            trajectories_by_session[session_id] = LineString(valid_probes)
        else:
            logging.warning(f"Session {session_id} has insufficient valid probes.")

    return trajectories_by_session

# Save Trajectories
def save_trajectories(trajectories_by_session):
    """
    Save the trajectories to a GeoDataFrame and export to GeoJSON.

    Args:
    - trajectories_by_session: Dictionary of trajectories by session ID.
    """
    trajectories_gdf = gpd.GeoDataFrame(
        {"session_id": list(trajectories_by_session.keys())},
        geometry=list(trajectories_by_session.values()),
        crs="EPSG:3857",
    )

    # Reproject the output back to EPSG:4326 for saving
    trajectories_gdf = trajectories_gdf.to_crs(epsg=4326)
    trajectories_gdf.to_file("trajectories_by_session.geojson", driver="GeoJSON")
    logging.info(f"Number of trajectories: {len(trajectories_by_session)}")
    logging.info("Trajectories saved to 'trajectories_by_session.geojson'")

if __name__ == "__main__":
    closed_road_path = "line_string.geojson"
    probe_data_path = "data/getafix_contruction_case_netherlands_after_1359704470_1729170279.csv"
    # probe_data_path = "data/getafix_contruction_case_netherlands_during_1359704470_1729170234.csv"

    # Load data
    closed_road_gdf, probe_data_gdf = load_data(closed_road_path, probe_data_path)

    # Process trajectories
    trajectories_by_session = process_trajectories(closed_road_gdf, probe_data_gdf)

    # Save trajectories
    save_trajectories(trajectories_by_session)

    if len(trajectories_by_session) > MIN_TRAJECTORIES:
        print("Final state: Road is OPEN.")
    else:
        print("Final state: Road is CLOSED (or not conclusively opened).")

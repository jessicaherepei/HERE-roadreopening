import geopandas as gpd
import pandas as pd
import math
from shapely.geometry import Point, LineString, box
from shapely.ops import transform
from tqdm import tqdm

# Configurations
DISTANCE_THRESHOLD = 1000.0  # Max distance in meters
DELTA_HEADING_MAX = 30.0  # Max heading difference in degrees

# Helper Functions
def distance_to_line(line, point):
    """Calculate distance from a point to a line."""
    return line.distance(point)

def compute_road_heading(segment, point):
    """Compute heading of the road at a given point."""
    if not isinstance(segment, LineString) or segment.is_empty:
        return None

    try:
        nearest_point = segment.interpolate(segment.project(point))
        coords = list(segment.coords)
        for i in range(len(coords) - 1):
            start, end = Point(coords[i]), Point(coords[i + 1])
            if start.equals(nearest_point) or end.equals(nearest_point):
                heading = math.degrees(math.atan2(end.y - start.y, end.x - start.x))
                return (heading + 360) % 360
    except Exception as e:
        print(f"Error in compute_road_heading: {e}")
    return None

def heading_difference(h1, h2):
    """Compute the absolute difference between two headings."""
    diff = abs(h1 - h2)
    return min(diff, 360 - diff)

def create_bounding_box(road_line, buffer_distance):
    """Create a bounding box with a given buffer around the road line."""
    road_bounds = road_line.bounds  # Get (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = road_bounds

    bounding_box = box(minx - buffer_distance, miny - buffer_distance, 
                        maxx + buffer_distance, maxy + buffer_distance)
    return bounding_box

closed_road_gdf = gpd.read_file("line_string.geojson").to_crs(epsg=4326)
probe_data_df = pd.read_csv("data\getafix_contruction_case_netherlands_during_1359704470_1729170234.csv")
probe_data_gdf = gpd.GeoDataFrame(
    probe_data_df,
    geometry=gpd.points_from_xy(probe_data_df.longitude, probe_data_df.latitude),
    crs="EPSG:4326",
).to_crs(epsg=4326)

coordinates = [list(geom.coords)[0] for geom in closed_road_gdf.geometry]
closed_road_line = LineString(coordinates)
buffer_distance = DISTANCE_THRESHOLD  # buffer in meters
bounding_box = create_bounding_box(closed_road_line, buffer_distance)

valid_probes = []

# Process and filter in one loop
for point in tqdm(probe_data_gdf.geometry, desc="Processing and filtering probe data"):
    # Check if the point is within the bounding box
    if bounding_box.contains(point):
    #     dist = distance_to_line(closed_road_line, point)
        
    #     if dist <= DISTANCE_THRESHOLD:  # Check distance threshold
        #segment_heading = compute_road_heading(closed_road_line, point)

        # Validate based on heading difference
        #road_heading = compute_road_heading(closed_road_line, point)
        #if segment_heading is not None and road_heading is not None:
         #   if heading_difference(road_heading, segment_heading) <= DELTA_HEADING_MAX:
        valid_probes.append((point))

# Group Valid Probes into Trajectories
trajectories = []
trajectory_points = []
for point in valid_probes:
    trajectory_points.append(point)
if trajectory_points:  
    trajectories.append(LineString(trajectory_points))

num_trajectories = len(trajectories)
print(f"Number of trajectories: {num_trajectories}")

trajectories_gdf = gpd.GeoDataFrame(
    geometry=trajectories,
    crs="EPSG:4326",
)
trajectories_gdf.to_file("trajectories.geojson", driver="GeoJSON")
print("Trajectories saved to 'trajectories.geojson'")

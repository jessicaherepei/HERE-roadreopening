import geopandas as gpd
import pandas as pd
import math
from shapely.geometry import Point, LineString, MultiPoint
from shapely.ops import split
from tqdm import tqdm

# Configurations
SEGMENT_LENGTH = 5  # Length of segments in meters
DISTANCE_THRESHOLD = 10.0  # Max distance in meters
DELTA_HEADING_MAX = 30.0  # Max heading difference in degrees
FRACTION_IN_USE_THRESHOLD = 0.5

# Helper Functions
# def subdivide_linestring(line, length):
#     """Split a LineString into segments of a specified length."""
#     if line.length <= length:
#         return [line]

#     segments = []
#     current_position = 0.0
#     while current_position < line.length:
#         next_position = min(current_position + length, line.length)
#         segment = line.interpolate(current_position).buffer(1e-9).intersection(line.interpolate(next_position).buffer(1e-9).union(line))
#         if isinstance(segment, LineString):
#             segments.append(segment)
#         current_position = next_position
#     print(segments)
#     return segments
def subdivide_linestring(line, length):
    """Split a LineString into segments of a specified length."""
    if line.length <= length:
        return [line]

    # Calculate division points
    num_divisions = int(line.length // length)
    division_points = [line.interpolate(length * i) for i in range(1, num_divisions + 1)]

    # Split the LineString
    result = split(line, MultiPoint(division_points))
    return list(result)

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

# Load Data
closed_road_gdf = gpd.read_file("line_string.geojson").to_crs(epsg=4326)
probe_data_df = pd.read_csv("during_filtered.csv")
probe_data_gdf = gpd.GeoDataFrame(
    probe_data_df,
    geometry=gpd.points_from_xy(probe_data_df.longitude, probe_data_df.latitude),
    crs="EPSG:4326",
).to_crs(epsg=4326)


# Extract the coordinates from the Point geometries
coordinates = [list(geom.coords)[0] for geom in closed_road_gdf.geometry]

# Create a LineString from the points
closed_road_line = LineString(coordinates)
print(closed_road_line)
road_segments = subdivide_linestring(closed_road_line, SEGMENT_LENGTH)
# Assign Probe Data to Segments
from tqdm import tqdm
import pandas as pd

# Initialize lists for results
segment_ids = []
probe_headings = []
valid_probes = []

# Process and filter in one loop
for point in tqdm(probe_data_gdf.geometry, desc="Processing and filtering probe data"):
    min_distance = float("inf")
    closest_segment = None

    # Find the closest segment
    for i, segment in enumerate(road_segments):
        if not isinstance(segment, LineString):
            continue  # Skip invalid segments
        dist = distance_to_line(segment, point)
        if dist < min_distance:
            min_distance = dist
            closest_segment = i

    # Check distance threshold
    if min_distance <= DISTANCE_THRESHOLD and closest_segment is not None:
        segment_heading = compute_road_heading(road_segments[closest_segment], point)
        segment_ids.append(closest_segment)
        probe_headings.append(segment_heading)

        # Validate based on heading difference
        road_heading = compute_road_heading(road_segments[closest_segment], point)
        if segment_heading is not None and road_heading is not None:
            if heading_difference(road_heading, segment_heading) <= DELTA_HEADING_MAX:
                valid_probes.append(closest_segment)
    else:
        segment_ids.append(None)
        probe_headings.append(None)


# Add results to GeoDataFrame
probe_data_gdf["segment_id"] = segment_ids
probe_data_gdf["probe_heading"] = probe_headings

segment_usage = pd.Series(valid_probes).value_counts(normalize=True)
segment_in_use = segment_usage[segment_usage >= FRACTION_IN_USE_THRESHOLD].index
# Output Results
print(f"Segments in use: {list(segment_in_use)}")
# Determine if the road is open or closed
total_segments = len(road_segments)
used_segments_count = len(segment_in_use)
fraction_in_use = used_segments_count / total_segments

road_status = "open" if fraction_in_use >= FRACTION_IN_USE_THRESHOLD else "closed"

# Output Results
print(f"Segments in use: {list(segment_in_use)}")
print(f"Road status: {road_status}")

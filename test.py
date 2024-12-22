import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString

# ---------------------------------------------------------------------
# 1. Configuration & Parameters
# ---------------------------------------------------------------------

# Distances in meters
DISTANCE_THRESHOLD = 10.0  

# Heading difference threshold (degrees)
DELTA_HEADING_MAX = 30.0  

# Minimum speed in m/s (about 10.8 km/h)
VELOCITY_MIN = 3.0  

# Subdivide the road every XX meters
SEGMENT_LENGTH = 50.0

# --- Session-based thresholds ---
# Minimum unique sessions to consider a segment "in use"
SEGMENT_SESSION_THRESHOLD = 2

# Fraction of segments that must be "in use" to declare the entire road open
FRACTION_IN_USE_THRESHOLD = 0.5

# ---------------------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------------------

def compute_heading_degrees(p1: Point, p2: Point) -> float:
    """
    Compute approximate heading in degrees from p1 to p2, 0 = North, 90 = East.
    (Typical "compass" heading, clockwise from North).
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle_radians = np.arctan2(dx, dy)
    angle_degrees = np.degrees(angle_radians)
    # Convert from mathematical angle (0 = x-axis, CCW) to compass angle (0 = north, CW).
    compass_angle = 90.0 - angle_degrees
    compass_angle %= 360.0
    return compass_angle

def minimal_heading_difference(h1: float, h2: float) -> float:
    """Return the minimal absolute difference between two headings (0..360)."""
    diff = abs(h1 - h2)
    return min(diff, 360.0 - diff)

def compute_road_heading(line: LineString, vehicle_point: Point, offset: float = 5.0) -> float:
    """
    Estimate the local road heading by:
      1. Projecting the vehicle_point onto the line.
      2. Interpolating a small offset forward along the line.
      3. Computing heading from the projection to the offset point.
    """
    proj_dist = line.project(vehicle_point)
    nearest_pt = line.interpolate(proj_dist)
    forward_proj = min(proj_dist + offset, line.length)
    forward_pt = line.interpolate(forward_proj)
    return compute_heading_degrees(nearest_pt, forward_pt)

def distance_to_line(point: Point, line: LineString) -> float:
    """
    Return the distance from a point to a given line (LineString).
    """
    return point.distance(line)

def subdivide_linestring(line: LineString, segment_length: float) -> list[LineString]:
    """
    Divide a LineString into smaller sub-linestrings each of approximately
    'segment_length' in length. This helps with partial usage detection.
    """
    segments = []
    dist_covered = 0.0
    while dist_covered < line.length:
        start_pt = line.interpolate(dist_covered)
        end_dist = min(dist_covered + segment_length, line.length)
        end_pt = line.interpolate(end_dist)
        segment_line = LineString([start_pt, end_pt])
        segments.append(segment_line)
        dist_covered += segment_length
    return segments

# ---------------------------------------------------------------------
# 3. Main Routine
# ---------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------
    # 3.1 Load the closed (or partially closed) road geometry
    # -----------------------------------------------------------------
    closed_road_gdf = gpd.read_file('closed_road.geojson')
    # Assume we only have 1 feature for the relevant road
    closed_road_line = closed_road_gdf.geometry.iloc[0]

    # -----------------------------------------------------------------
    # 3.2 Load raw probe data (CSV)
    # -----------------------------------------------------------------
    probe_data_df = pd.read_csv('./data/raw_probe_direction_of_travel_eindhoven_20240117_20240118.csv')
    # Expected columns: ['lat', 'lon', 'heading', 'speed', 'session_id']
    # (We no longer need 'timestamp' or 'datetime'.)

    # Convert to GeoDataFrame
    probe_data_gdf = gpd.GeoDataFrame(
        probe_data_df,
        geometry=gpd.points_from_xy(probe_data_df.lon, probe_data_df.lat),
        crs='EPSG:4326'
    )
    closed_road_gdf = closed_road_gdf.to_crs(epsg=4326)
    probe_data_gdf = probe_data_gdf.to_crs(epsg=4326)
    # Update reference to the line in projected coords
    closed_road_line = closed_road_gdf.geometry.iloc[0]

    # -----------------------------------------------------------------
    # 4. Subdivide the Road for Partial Detection
    # -----------------------------------------------------------------
    road_segments = subdivide_linestring(closed_road_line, SEGMENT_LENGTH)

    # -----------------------------------------------------------------
    # 5. Basic Map Matching & Filtering
    # -----------------------------------------------------------------
    # For each point, find the nearest sub-segment and filter by:
    #   - distance threshold
    #   - heading threshold
    #   - minimum speed
    segment_ids = []
    distances = []
    headings_road = []

    for idx, row in probe_data_gdf.iterrows():
        pt = row.geometry
        # Find nearest segment
        best_seg_id = None
        best_dist = float('inf')
        best_seg = None
        for seg_id, seg_line in enumerate(road_segments):
            d = distance_to_line(pt, seg_line)
            if d < best_dist:
                best_dist = d
                best_seg_id = seg_id
                best_seg = seg_line

        segment_ids.append(best_seg_id)
        distances.append(best_dist)

        # Road heading at that point
        local_heading = compute_road_heading(best_seg, pt) if best_seg else np.nan
        headings_road.append(local_heading)

    probe_data_gdf['road_segment_id'] = segment_ids
    probe_data_gdf['dist_to_road'] = distances
    probe_data_gdf['road_heading'] = headings_road

    # Distance filter
    probe_data_gdf = probe_data_gdf[probe_data_gdf['dist_to_road'] <= DISTANCE_THRESHOLD].copy()

    # Heading filter
    def heading_filter(row):
        hdg_diff = minimal_heading_difference(row['heading'], row['road_heading'])
        return hdg_diff <= DELTA_HEADING_MAX

    probe_data_gdf = probe_data_gdf[probe_data_gdf.apply(heading_filter, axis=1)].copy()

    # Speed filter
    probe_data_gdf = probe_data_gdf[probe_data_gdf['speed'] >= VELOCITY_MIN].copy()

    # -----------------------------------------------------------------
    # 6. Count Session Usage on Each Segment
    # -----------------------------------------------------------------
    # We group by "road_segment_id" and count the number of unique session_ids.
    segment_usage = probe_data_gdf.groupby('road_segment_id')['session_id'].nunique()

    # A segment is considered "in use" if the # of sessions >= SEGMENT_SESSION_THRESHOLD
    segment_in_use = (segment_usage >= SEGMENT_SESSION_THRESHOLD)

    # Compute how many segments are in use
    num_segments_in_use = segment_in_use.sum()
    total_segments = len(segment_in_use)
    fraction_in_use = num_segments_in_use / total_segments if total_segments > 0 else 0.0

    # -----------------------------------------------------------------
    # 7. Decide if Road is Open or Closed
    # -----------------------------------------------------------------
    # If >= 50% of the segments are in use (example), we say "open"
    # You can adjust FRACTION_IN_USE_THRESHOLD to your needs.
    if fraction_in_use >= FRACTION_IN_USE_THRESHOLD:
        print("Road is OPEN based on session usage.")
    else:
        print("Road is CLOSED (or not conclusively open).")

    # For debugging, you can print more details:
    print(f"Number of segments in use: {num_segments_in_use}/{total_segments}")
    print(f"Fraction in use: {fraction_in_use:.2f}")

if __name__ == "__main__":
    main()

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import datetime

# ----------------------------------------------------
# 1. Configuration & Advanced Parameters
# ----------------------------------------------------

# Distances in meters
DISTANCE_THRESHOLD = 10.0  

# Heading difference threshold (degrees)
DELTA_HEADING_MAX = 30.0  

# Minimum speed in m/s (about 10.8 km/h)
VELOCITY_MIN = 3.0  

# For partial segment approach, subdivide the road every XX meters
SEGMENT_LENGTH = 50.0  

# Time bin size (for the rolling or binned approach)
TIME_WINDOW = '5min'

# Minimum unique trajectories in a time bin to consider the road "in use"
TRAJECTORY_THRESHOLD = 5  

# Number of consecutive bins meeting the threshold to declare "reopened"
CONSECUTIVE_BINS_OPEN = 3  

# Number of consecutive bins below threshold to declare "closed" again
CONSECUTIVE_BINS_CLOSED = 3  

# Rolling window size (number of bins) for smoothing usage counts
ROLLING_WINDOW_SIZE = 3  

# Outlier detection thresholds
MAX_HEADING_CHANGE_PER_SEC = 40.0  # deg/sec, just an example
MAX_SPEED_CHANGE_PER_SEC = 5.0     # m/s^2, e.g., 5 m/s per second


# ----------------------------------------------------
# 2. Helper Functions
# ----------------------------------------------------

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
    """ Return the minimal absolute difference between two headings (0..360) """
    diff = abs(h1 - h2)
    return min(diff, 360.0 - diff)

def compute_road_heading(line: LineString, vehicle_point: Point, offset: float = 5.0) -> float:
    """
    Estimate the local road heading by:
      1. Finding the projection of vehicle_point onto the line.
      2. Interpolating a small offset forward along the line.
      3. Computing heading from the projection to the offset.
    """
    proj_dist = line.project(vehicle_point)
    nearest_pt = line.interpolate(proj_dist)
    forward_proj = min(proj_dist + offset, line.length)
    forward_pt = line.interpolate(forward_proj)
    return compute_heading_degrees(nearest_pt, forward_pt)

def distance_to_line(point: Point, line: LineString) -> float:
    return point.distance(line)

def detect_outliers_in_trajectory(traj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional step to remove outlier points within a trajectory based on
    unrealistic heading or speed changes between consecutive points.
    
    This is a simplistic approach:
      - Sort by timestamp
      - Check heading/speed changes vs. time difference
      - Drop points that exceed thresholds
    """
    traj_df = traj_df.sort_values('sample_date')
    to_drop = []
    prev_row = None
    for idx, row in traj_df.iterrows():
        if prev_row is not None:
            dt = (row['sample_date'] - prev_row['sample_date']).total_seconds()
            if dt > 0:
                heading_change = minimal_heading_difference(row['heading'], prev_row['heading'])
                speed_change = abs(row['speed'] - prev_row['speed'])
                
                # heading change per second
                hcps = heading_change / dt
                # speed change per second
                scps = speed_change / dt
                
                if hcps > MAX_HEADING_CHANGE_PER_SEC or scps > MAX_SPEED_CHANGE_PER_SEC:
                    # Mark this point as outlier
                    to_drop.append(idx)
        prev_row = row
    return traj_df.drop(index=to_drop)

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


# ----------------------------------------------------
# 3. Load Road Geometry & Probe Data
# ----------------------------------------------------

def main():
    # 3.1 Load the closed (or partially closed) road geometry
    closed_road_gdf = gpd.read_file('closed_road.geojson')
    # Suppose it has only 1 feature for the relevant road
    closed_road_line = closed_road_gdf.geometry.iloc[0]
    
    # 3.2 Load raw probe data
    probe_data_df = pd.read_csv('probe_data.csv')
    # Expected columns: ['lat', 'lon', 'heading', 'speed', 'trajectory_id', 'timestamp']
    probe_data_df['sample_date'] = pd.to_datetime(probe_data_df['sample_date'])
    
    # Convert to GeoDataFrame
    probe_data_gdf = gpd.GeoDataFrame(
        probe_data_df,
        geometry=gpd.points_from_xy(probe_data_df.lon, probe_data_df.lat),
        crs='EPSG:4326'  # or whichever CRS your data is in
    )
    
    # Reproject to a projected coordinate system for accurate distance
    closed_road_gdf = closed_road_gdf.to_crs(epsg=3857)
    probe_data_gdf = probe_data_gdf.to_crs(epsg=3857)
    
    # Update reference to the line in projected coordinates
    closed_road_line = closed_road_gdf.geometry.iloc[0]
    
    
    # ----------------------------------------------------
    # 4. Optional: Subdivide the Road for Partial Detection
    # ----------------------------------------------------
    road_segments = subdivide_linestring(closed_road_line, SEGMENT_LENGTH)
    
    
    # ----------------------------------------------------
    # 5. Basic "Map Matching" & Filtering
    # ----------------------------------------------------
    # We'll do a simplified approach:
    # 1. For each point, find the nearest line segment among the sub-segments
    # 2. Filter by distance threshold, heading threshold, and speed
    
    # If you prefer to do it on the entire line, you can skip subdividing or
    # do your own advanced map matching approach.
    
    # We'll store the best (sub-segment) match info in new columns:
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
        
        # Store results
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
    
    
    # ----------------------------------------------------
    # 6. (Optional) Outlier Detection Per Trajectory
    # ----------------------------------------------------
    filtered_dfs = []
    for traj_id, subdf in probe_data_gdf.groupby('session_id'):
        clean_subdf = detect_outliers_in_trajectory(subdf)
        filtered_dfs.append(clean_subdf)
    probe_data_gdf = pd.concat(filtered_dfs).sort_values(['session_id', 'sample_date'])
    
    
    # ----------------------------------------------------
    # 7. Usage Computation: Rolling or Binned
    # ----------------------------------------------------
    # Let's bin data into TIME_WINDOW intervals, count distinct trajectories
    probe_data_gdf['time_bin'] = probe_data_gdf['sample_date'].dt.floor(TIME_WINDOW)
    
    # Count distinct trajectories per time bin
    usage_series = probe_data_gdf.groupby('time_bin')['session_id'].nunique().sort_index()
    
    # Optionally apply a rolling mean to smooth out noise
    usage_smoothed = usage_series.rolling(ROLLING_WINDOW_SIZE, min_periods=1).mean()
    
    # ----------------------------------------------------
    # 8. Partial Reopening: Evaluate Usage per Segment
    # ----------------------------------------------------
    # For advanced analysis, we can also do a pivot: time_bin x road_segment_id -> unique trajectories
    segment_usage = probe_data_gdf.groupby(['time_bin', 'road_segment_id'])['session_id'].nunique().unstack(fill_value=0)
    # segment_usage is now a DataFrame: rows = time_bins, columns = segment IDs
    
    # We could define a threshold for each segment:
    # e.g. consider a segment "in use" if >= 3 distinct trajectories in the bin
    SEGMENT_TRAJ_THRESHOLD = 3
    
    # We can then see how many segments are "in use" for each time bin
    segment_in_use_count = (segment_usage >= SEGMENT_TRAJ_THRESHOLD).sum(axis=1)
    # Or the fraction of segments in use:
    fraction_in_use = segment_in_use_count / len(road_segments)
    
    
    # ----------------------------------------------------
    # 9. Advanced Decision Logic with Hysteresis
    # ----------------------------------------------------
    # We want to declare the entire road open if the overall usage
    # (based on usage_smoothed) is above TRAJECTORY_THRESHOLD for
    # CONSECUTIVE_BINS_OPEN bins, or if a significant fraction of
    # segments are in use. Then close it again if usage is consistently below.

    # Maintain a state: "closed", "opening", "open", "closing" 
    # for demonstration, keep it simple with "open"/"closed" plus counters.
    current_state = "closed"
    consecutive_open_bins = 0
    consecutive_closed_bins = 0
    
    for time_bin, smoothed_count in usage_smoothed.items():
        # Check fraction of segments in use
        frac_in_use = fraction_in_use.loc[time_bin] if time_bin in fraction_in_use.index else 0.0
        
        # Decide if we meet "open" criteria in this bin
        # We combine two conditions: 
        # 1) smoothed trajectory count >= threshold
        # 2) fraction of segments in use > 50% (example)
        is_open_bin = (smoothed_count >= TRAJECTORY_THRESHOLD) and (frac_in_use >= 0.5)
        
        if is_open_bin:
            consecutive_open_bins += 1
            consecutive_closed_bins = 0
        else:
            consecutive_open_bins = 0
            consecutive_closed_bins += 1
        
        # Evaluate transitions
        if current_state == "closed" and consecutive_open_bins >= CONSECUTIVE_BINS_OPEN:
            current_state = "open"
            print(f"{time_bin}: The road appears to have reopened.")
        
        elif current_state == "open" and consecutive_closed_bins >= CONSECUTIVE_BINS_CLOSED:
            current_state = "closed"
            print(f"{time_bin}: The road appears to have closed again.")
    
    if current_state == "open":
        print("Final state: Road is OPEN.")
    else:
        print("Final state: Road is CLOSED (or not conclusively opened).")

    
if __name__ == "__main__":
    main()
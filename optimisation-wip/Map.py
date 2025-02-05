import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union, nearest_points
from shapely.geometry import Point
import random
import os
from bayes_opt import BayesianOptimization
from functools import lru_cache

# Update the config settings
ox.settings.use_cache = True
ox.settings.log_console = True

# Filepath for the saved graph
graph_filepath = "Bris_Region.graphml"

def load_wpc_data():
    """Load and combine England and Wales WPC data"""
    print("\nLoading England and Wales WPC data...")
    try:
        england_wpc = gpd.read_file("England_WPC.json")
        wales_wpc = gpd.read_file("Wales_WPC.json")
        # Combine the datasets
        return pd.concat([england_wpc, wales_wpc], ignore_index=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure both England_WPC.json and Wales_WPC.json exist in the data directory")
        return None

constituencies = [
    'Bristol West',
    'Bristol South',
    'Bristol East',
    'Bristol North West',
    'Filton and Bradley Stoke',
    'North Somerset',
    'Weston-Super-Mare',
    'North East Somerset',
    'Bath',
    'Kingswood',
    'Thornbury and Yate',
    'Stroud',
    'Gloucester',
    'Forest of Dean',
    'Monmouth',
    'Newport East',
    'Newport West'
]

def get_bristol_network(boundary_shape):
    """Get the road network within the boundary shape"""
    G = ox.graph_from_polygon(boundary_shape, network_type='drive')
    G = ox.add_edge_speeds(G)  # Add edge speeds
    G = ox.add_edge_travel_times(G)  # Add edge travel times
    return G

# Check if the graph file exists
if os.path.exists(graph_filepath):
    # Load the graph from file
    print("Loading graph from file...")
    G = ox.load_graphml(graph_filepath)
    # Since boundary_shape is not saved with the graph, recalc a fallback boundary using nodes
    nodes, _ = ox.graph_to_gdfs(G)
    boundary_shape = nodes.unary_union.convex_hull
else:
    # Create and save the graph
    print("Downloading and creating graph...")
    
    # Load and combine WPC data
    combined_wpc = load_wpc_data()
    if combined_wpc is None:
        raise Exception("Failed to load WPC data")
    
    # Filter for our region
    bristol_region = combined_wpc[combined_wpc['PCON13NM'].isin(constituencies)].copy()

    print(f"Found {len(bristol_region)} constituencies in the region")

    # Clean and create boundary
    print("\nCleaning and creating Bristol region boundary...")
    bristol_region.geometry = bristol_region.geometry.buffer(0)
    bristol_region = bristol_region[bristol_region.geometry.is_valid]
    boundary_shape = unary_union(bristol_region.geometry)
    
    # Get the network using the boundary
    print("Downloading road network...")
    G = get_bristol_network(boundary_shape)
    
    # Save the graph
    ox.save_graphml(G, filepath=graph_filepath)
    print("Graph saved for future use.")

## Load constituency population data and create a population density GeoDataFrame
try:
    pop_df = pd.read_csv("constituency_populations.csv")
    # Assumes the CSV file has columns "constituency" and "population"
    # Reload the WPC data to access constituency boundaries (if not available already)
    combined_wpc = load_wpc_data()
    if combined_wpc is not None:
         # Filter for the region; this uses the same column as before, e.g. "PCON13NM"
         bristol_region = combined_wpc[combined_wpc['PCON13NM'].isin(constituencies)].copy()
         # Merge the population data using matching constituency names
         pop_density_gdf = bristol_region.merge(pop_df, left_on="PCON13NM", right_on="constituency", how="left")
         # Reproject to a metric CRS (e.g., EPSG:27700 for the UK) for proper area calculations
         pop_density_gdf = pop_density_gdf.to_crs(epsg=27700)
         pop_density_gdf["area"] = pop_density_gdf.geometry.area
         pop_density_gdf["density"] = pop_density_gdf["population"] / pop_density_gdf["area"]
         # Reproject back to EPSG:4326 to match the graph's CRS
         pop_density_gdf = pop_density_gdf.to_crs(epsg=4326)
         print("Population density GeoDataFrame created successfully.")
except Exception as e:
    print("Error processing constituency population data:", e)

# Load garage coordinates
def load_garage_coords():
    """Load garage coordinates from filtered CSV file"""
    print("Loading garage coordinates...")
    try:
        garages_df = pd.read_csv('filtered_garages.csv')
        # Convert to GeoDataFrame
        geometry = [Point(xy) for xy in zip(garages_df.Longitude, garages_df.Latitude)]
        garages_gdf = gpd.GeoDataFrame(garages_df, geometry=geometry, crs="EPSG:4326")
        return garages_gdf
    except Exception as e:
        print(f"Error loading garage coordinates: {e}")
        return None

# Load garages
garages_gdf = load_garage_coords()

# Define the garages to move
garages_to_move = ['GL2 5DQ', 'BS5 6NJ']

def to_valid_point(point, boundary, projection_penalty_factor=500):
    """
    If point is within boundary, return it unchanged and zero penalty.
    Otherwise, project the point to the nearest location on the boundary,
    and return the projected point along with a penalty proportional to the projection movement.
    """
    if boundary.contains(point):
        return point, 0
    else:
        # nearest_points returns a tuple (point on boundary, the original point's projection)
        projected_point = nearest_points(boundary, point)[0]
        penalty = projection_penalty_factor * point.distance(projected_point)
        return projected_point, penalty

# Function to calculate average response time based on garage locations
def average_response_time(x1, y1, x2, y2):
    # Create Point objects for the candidate garage locations
    point1 = Point(x1, y1)
    point2 = Point(x2, y2)

    # Project candidate points into the boundary if necessary and get a projection penalty.
    point1_valid, penalty1 = to_valid_point(point1, boundary_shape)
    point2_valid, penalty2 = to_valid_point(point2, boundary_shape)
    projection_penalty = penalty1 + penalty2

    # (Optional) Additional penalty if too close to the boundary.
    inside_penalty = 0
    inside_penalty_threshold = 0.005  # Adjust as needed
    inside_penalty_factor = 100       # Adjust as needed

    if boundary_shape.contains(point1_valid):
        d_edge = point1_valid.distance(boundary_shape.boundary)
        if d_edge < inside_penalty_threshold:
            inside_penalty += inside_penalty_factor * (inside_penalty_threshold - d_edge)
    if boundary_shape.contains(point2_valid):
        d_edge = point2_valid.distance(boundary_shape.boundary)
        if d_edge < inside_penalty_threshold:
            inside_penalty += inside_penalty_factor * (inside_penalty_threshold - d_edge)

    total_penalty = projection_penalty + inside_penalty

    garages_gdf_copy = garages_gdf.copy()
    # Update garage locations using the valid (projected) Point objects
    garages_gdf_copy.loc[garages_gdf_copy['Postcode'] == garages_to_move[0], 'geometry'] = point1_valid
    garages_gdf_copy.loc[garages_gdf_copy['Postcode'] == garages_to_move[1], 'geometry'] = point2_valid

    # Reuse the precomputed breakdown points
    breakdowns_gdf = precomputed_breakdowns

    # Calculate average response time (note: find_closest_garages returns negative avg response time)
    base_response = find_closest_garages(breakdowns_gdf, garages_gdf_copy, G)

    # Return the adjusted response time (base value minus penalty, so lower is worse)
    return base_response - total_penalty

def generate_random_breakdowns(num_points=100, jitter=0.0001, population_weight_factor=2.0):
    """Generate breakdown points along road segments weighted by edge length and population density.
    
    This method converts the graph to its edges GeoDataFrame. Each road segment's weight is computed
    using its length and, if available, the population density at its midpoint. Roads in high-density 
    areas are given extra weight based on the population_weight_factor parameter. Finally, a random point 
    is chosen along each sampled segment (with an optional jitter) to spread overlapping points.
    """
    # Get edges as a GeoDataFrame from the graph G (each edge is a road segment)
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    
    # Ensure each edge has a 'length' attribute (if not, compute it from the geometry)
    if 'length' not in edges_gdf.columns:
        edges_gdf['length'] = edges_gdf.geometry.length
    
    # If a population density GeoDataFrame exists and has a "density" column, merge density info with the edges.
    if 'pop_density_gdf' in globals() and 'density' in pop_density_gdf.columns:
        # Compute midpoints for each edge
        edges_gdf['midpoint'] = edges_gdf.geometry.centroid
        # Spatially join to assign density from pop_density_gdf (which should have a 'density' column)
        edges_gdf = gpd.sjoin(edges_gdf, pop_density_gdf[['density', 'geometry']], how='left', predicate='intersects')
        # Replace missing density values with zero
        edges_gdf['density'] = edges_gdf['density'].fillna(0)
        # Compute a new weight that factors in population density:
        # Longer roads and roads in dense areas get a higher weight.
        edges_gdf['weight'] = edges_gdf['length'] * (1 + population_weight_factor * edges_gdf['density'])
    else:
        edges_gdf['weight'] = edges_gdf['length']
    
    # Sample edges weighted by the computed weight with replacement
    sampled_edges = edges_gdf.sample(n=num_points, weights='weight', random_state=22, replace=True)
    
    breakdown_points = []
    for idx, edge in sampled_edges.iterrows():
        line = edge.geometry
        # Choose a random fraction along the length of the road segment
        fraction = random.random()
        point = line.interpolate(fraction * line.length)
        # Add a small jitter to further separate overlapping points
        jittered_point = Point(
            point.x + random.uniform(-jitter, jitter),
            point.y + random.uniform(-jitter, jitter)
        )
        breakdown_points.append(jittered_point)
    
    return gpd.GeoDataFrame(geometry=breakdown_points, crs="EPSG:4326")

# Cache repeated calls to find the nearest node from a given coordinate pair.
@lru_cache(maxsize=None)
def cached_nearest_node(x, y):
    # Using a tuple of coordinates as the key (floats are hashable)
    return ox.distance.nearest_nodes(G, x, y)

# Cache shortest path length between two nodes since many evaluations may reuse the same pair.
@lru_cache(maxsize=None)
def cached_shortest_path_length(u, v):
    return nx.shortest_path_length(G, u, v, weight='travel_time')

def find_closest_garages(breakdowns_gdf, garages_gdf, G):
    """Find the closest garage to each breakdown using Euclidean and network distance"""
    response_times = []
    
    for breakdown in breakdowns_gdf.geometry:
        # Calculate Euclidean distances to all garages
        garages_gdf['euclidean_distance'] = garages_gdf.geometry.apply(lambda g: breakdown.distance(g))
        
        # Get the three closest garages by Euclidean distance
        closest_garages = garages_gdf.nsmallest(3, 'euclidean_distance')
        
        # Cache breakdown node calculation
        breakdown_node = cached_nearest_node(breakdown.x, breakdown.y)
        
        min_travel_time = float('inf')
        closest_garage = None
        
        for _, garage in closest_garages.iterrows():
            garage_node = cached_nearest_node(garage.geometry.x, garage.geometry.y)
            try:
                # Compute the full shortest path from the breakdown to the garage
                path = nx.shortest_path(G, breakdown_node, garage_node, weight='travel_time')

                # Check if all nodes along the path are inside the boundary
                if not all(boundary_shape.contains(Point(G.nodes[node]['x'], G.nodes[node]['y'])) for node in path):
                    continue

                # Since the path is valid, get its travel time using our cached function
                travel_time_seconds = cached_shortest_path_length(breakdown_node, garage_node)
                travel_time_minutes = travel_time_seconds / 60  # Convert seconds to minutes
                if travel_time_minutes < min_travel_time:
                    min_travel_time = travel_time_minutes
                    closest_garage = garage
            except nx.NetworkXNoPath:
                continue
        
        if closest_garage is not None:
            # Calculate response time (time to scene * 2 + 30 mins repair time)
            response_time = (min_travel_time * 2)
            response_times.append(response_time)
    
    # Calculate average response time
    average_response_time = sum(response_times) / len(response_times) if response_times else float('inf')
    print(f"Average response time: {average_response_time:.2f} minutes")
    return -average_response_time

# Precompute breakdown points once (using desired parameters) and reuse them globally
precomputed_breakdowns = generate_random_breakdowns(num_points=200, jitter=0.0001, population_weight_factor=1.0)

# Bayesian Optimization to find optimal garage locations
def optimize_garage_locations():
    # Dynamically compute network bounds from the nodes in graph G
    nodes, _ = ox.graph_to_gdfs(G)
    minx, miny, maxx, maxy = nodes.total_bounds

    # Define pbounds for the garage locations covering the entire network area
    pbounds = {
        'x1': (minx, maxx),  # Garage 1 X-coordinate
        'y1': (miny, maxy),  # Garage 1 Y-coordinate
        'x2': (minx, maxx),  # Garage 2 X-coordinate
        'y2': (miny, maxy)   # Garage 2 Y-coordinate
    }

    # Increase the sample size and iterations for better exploration
    optimizer = BayesianOptimization(
        f=average_response_time,
        pbounds=pbounds,
        random_state=42
    )

    optimizer.maximize(init_points=10, n_iter=40)

    # Snap the optimal locations to the nearest node on the network and update to their coordinates
    optimal_locations = optimizer.max['params']
    node1 = ox.distance.nearest_nodes(G, optimal_locations['x1'], optimal_locations['y1'])
    node2 = ox.distance.nearest_nodes(G, optimal_locations['x2'], optimal_locations['y2'])
    
    # Retrieve the coordinates from the nearest nodes
    optimal_locations['x1'] = G.nodes[node1]['x']
    optimal_locations['y1'] = G.nodes[node1]['y']
    optimal_locations['x2'] = G.nodes[node2]['x']
    optimal_locations['y2'] = G.nodes[node2]['y']

    return optimal_locations  # Return the best found locations as valid coordinate pairs

# Run the optimization
optimal_locations = optimize_garage_locations()
print(f"Optimal garage locations: {optimal_locations}")

# Plot the updated garage locations
fig, ax = ox.plot_graph(G, node_size=0, edge_linewidth=0.5, show=False)

if garages_gdf is not None:
    garages_gdf.plot(ax=ax, color='yellow', marker='o', markersize=10, label='Initial Garages')

# Plot the relocated garage locations
for i in range(len(garages_to_move)):
    x = optimal_locations[f'x{i + 1}']
    y = optimal_locations[f'y{i + 1}']
    ax.plot(x, y, 'go', markersize=10, label='Relocated Garages')  # Plot relocated garages as green dots

# Generate breakdowns and plot them
breakdowns_gdf = generate_random_breakdowns()
breakdowns_gdf.plot(ax=ax, color='red', marker='x', markersize=10, label='Breakdowns')

plt.legend()
plt.show()



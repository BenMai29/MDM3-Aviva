import osmnx as ox
import networkx as nx
import timeit
import os
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def euclidean_heuristic(u, v):
    # Calculate straight-line distance between nodes u and v
    u_coord = G.nodes[u]['x'], G.nodes[u]['y']
    v_coord = G.nodes[v]['x'], G.nodes[v]['y']
    return ox.distance.euclidean_dist_vec(*u_coord, *v_coord)

# Configure OSMnx
ox.config(use_cache=True, log_console=True)

# Filepath for the saved graph
graph_filepath = "uk_highways.graphml"

# Check if the graph file exists
if os.path.exists(graph_filepath):
    # Load the graph from file
    print("Loading graph from file...")
    G = ox.load_graphml(graph_filepath)
else:
    # Create and save the graph
    print("Downloading and creating graph...")
    cf = '["highway"~"motorway|motorway_link|trunk|trunk_link"]'
    G = ox.graph_from_place(
        "United Kingdom",
        network_type="drive",
        custom_filter=None,
        simplify=True,
        retain_all=False,
    )
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    ox.save_graphml(G, filepath=graph_filepath)
    print("Graph saved for future use.")

# Geocode source and target addresses
source_address = "Bristol"
target_address = "edinburgh"

source = ox.geocode(source_address)
target = ox.geocode(target_address)

# Find nearest nodes
source_node = ox.nearest_nodes(G, source[1], source[0])
target_node = ox.nearest_nodes(G, target[1], target[0])

# Define timing functions
def compute_dijkstra():
    return nx.shortest_path(G, source_node, target_node, weight="travel_time", method="dijkstra")

def compute_astar():
    return nx.astar_path(G, source_node, target_node, weight="travel_time", heuristic=euclidean_heuristic)

# Time Dijkstra
dijkstra_time = timeit.timeit(compute_dijkstra, number=1)
print(f"{dijkstra_time:.4f} seconds to run Dijkstra")

# Time A*
astar_time = timeit.timeit(compute_astar, number=1)
print(f"{astar_time:.4f} seconds to run A*")

# Compute the routes
route = compute_dijkstra()
route2 = compute_astar()

# Plot both routes on the graph
fig, ax = ox.plot_graph(G, bgcolor="k", node_size=0, edge_linewidth=0.5, show=False, close=False)

# Overlay Dijkstra route
ox.plot_graph_route(G, route, route_linewidth=4, route_color="red", ax=ax, show=False, close=False)

# Overlay A* route
ox.plot_graph_route(G, route2, route_linewidth=4, route_color="blue", ax=ax, show=False, close=False)

# Add title and show the plot
ax.set_title("Routes: Dijkstra (red) vs A* (blue)", color="white")
plt.show()

# Load postcodes from CSV (Column F)
csv_file = "garagelocations.csv"
df = pd.read_csv(csv_file)

# Clean postcodes: strip spaces and uppercase
postcodes = df.iloc[:, 5].dropna().str.strip().str.upper().unique()

# List to store failed postcodes
failed_postcodes = []

# Geocode with retry logic
def safe_geocode(postcode, retries=2, delay=1):
    for attempt in range(retries):
        try:
            return postcode, ox.geocode(postcode)
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                failed_postcodes.append(postcode)  # Record failed postcode
                return postcode, None

# Multithreaded geocoding
def batch_geocode(postcodes, max_workers=20):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(safe_geocode, pc): pc for pc in postcodes}
        for future in as_completed(futures):
            pc, coord = future.result()
            if coord:
                results[pc] = coord
    return results

# Run batch geocoding
postcode_coords_dict = batch_geocode(postcodes)
postcode_coords = list(postcode_coords_dict.values())

# Find nearest nodes
postcode_nodes = [ox.nearest_nodes(G, lon, lat) for lat, lon in postcode_coords]

# Plot the graph and routes
fig, ax = ox.plot_graph(G, bgcolor="k", node_size=0, edge_linewidth=0.5, show=False, close=False)

# Plot Dijkstra route in red
ox.plot_graph_route(G, route, route_linewidth=4, route_color="red", ax=ax, show=False, close=False)

# Plot A* route in blue
ox.plot_graph_route(G, route2, route_linewidth=4, route_color="blue", ax=ax, show=False, close=False)

# Plot postcode markers
x_coords = [G.nodes[node]['x'] for node in postcode_nodes]
y_coords = [G.nodes[node]['y'] for node in postcode_nodes]
ax.scatter(x_coords, y_coords, c="yellow", s=10, marker="o", label="Postcodes")

# Add title and legend
ax.set_title("Routes and Postcodes", color="white")
ax.legend(facecolor="gray")

plt.show()

# Print postcodes that failed to geocode
if failed_postcodes:
    print("\nPostcodes that failed to geocode:")
    for pc in failed_postcodes:
        print(pc)
else:
    print("\nAll postcodes were successfully geocoded!")


import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import json
import os
from scipy.spatial import Voronoi
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import contextily as ctx

# Configure OSMnx settings
ox.settings.log_console = True
ox.settings.use_cache = True

def get_uk_network():
    """Get UK road network, loading from cache if available"""
    network_path = 'data/uk_network.graphml'

    if os.path.exists(network_path):
        print("Loading UK road network from cache...")
        G = ox.load_graphml(network_path)

        first_edge = list(G.edges(data=True))[0]
        if 'travel_time' not in first_edge[2]:
            print("Adding edge speeds and travel times...")
            G = ox.routing.add_edge_speeds(G)
            G = ox.routing.add_edge_travel_times(G)
            ox.save_graphml(G, network_path)
    else:
        print("Downloading UK road network...")
        G = ox.graph_from_place("United Kingdom",
                               network_type="drive",
                               simplify=True)
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)
        os.makedirs('data', exist_ok=True)
        ox.save_graphml(G, network_path)

    return G

def create_voronoi_regions(garages):
    """Create Voronoi regions for garages"""
    points = np.array([[float(g['lng']), float(g['lat'])] for g in garages['garageMarkers']])
    vor = Voronoi(points)
    return vor

def get_nodes_in_region(G, garage_coords, vor_region):
    """Get nodes that fall within a Voronoi region"""
    nodes_in_region = []
    for node, data in G.nodes(data=True):
        node_coords = [data['x'], data['y']]
        if point_in_polygon(node_coords, vor_region):
            nodes_in_region.append(node)
    return nodes_in_region

def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def process_garage(args):
    """Process a single garage (for parallel execution)"""
    G, garage, nodes_subset, max_time = args
    try:
        lat = float(garage['lat'])
        lng = float(garage['lng'])
        name = garage['companyName']

        center_node = ox.nearest_nodes(G, lng, lat)

        # Only calculate times for nodes in this garage's region
        travel_times = nx.single_source_dijkstra_path_length(
            G.subgraph(nodes_subset),
            center_node,
            weight='travel_time'
        )

        travel_times = {node: time/60 for node, time in travel_times.items()}

        nodes_to_plot = []
        node_colors = []
        for node in nodes_subset:
            time = travel_times.get(node, float('inf'))
            if time <= max_time:
                nodes_to_plot.append(node)
                node_colors.append(time)

        return {
            'nodes': nodes_to_plot,
            'colors': node_colors,
            'garage_coords': (lng, lat),
            'name': name
        }
    except Exception as e:
        print(f"Error processing garage {name}: {e}")
        return None

def create_isochrone_plot(G, garages, max_time=60, filename='uk_garage_isochrones_sample.png'):
    # Create Voronoi regions
    vor = create_voronoi_regions(garages)

    # Prepare arguments for parallel processing
    process_args = []
    for i, garage in enumerate(garages['garageMarkers']):
        region_vertices = vor.vertices[vor.regions[vor.point_region[i]]]
        nodes_subset = get_nodes_in_region(G, garage, region_vertices)
        process_args.append((G, garage, nodes_subset, max_time))

    # Process garages in parallel
    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_garage, args) for args in process_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing garages"):
            result = future.result()
            if result:
                results.append(result)

    # Create plot
    fig, ax = plt.subplots(figsize=(20, 20))

    # Updated color scheme
    colors = ['#67000d',  # Dark red
             '#ef3b2c',   # Medium red
             '#fc9272',   # Light red
             '#fee0d2']   # Very light red
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

    # Plot base network (fainter)
    ox.plot_graph(G, ax=ax, node_size=0, edge_color='#eeeeee',
                 edge_linewidth=0.3, show=False)

    # Plot Voronoi regions
    for region in vor.regions:
        if -1 not in region and len(region) > 0:  # Valid region
            # Get vertices for this region
            polygon = vor.vertices[region]
            # Close the polygon by repeating first vertex
            polygon = np.vstack((polygon, polygon[0]))
            # Plot the polygon boundary
            ax.plot(polygon[:, 0], polygon[:, 1],
                   color='blue', linewidth=0.5,
                   alpha=0.5, linestyle='--',
                   zorder=4)

    # Plot results
    for result in results:
        nodes = result['nodes']
        node_colors = result['colors']

        # Get all edges connected to these nodes
        edges_to_plot = []
        edge_colors = []

        for u, v, data in G.edges(data=True):
            if u in nodes and v in nodes:
                # Get average travel time of connected nodes
                u_time = next((c for n, c in zip(nodes, node_colors) if n == u), None)
                v_time = next((c for n, c in zip(nodes, node_colors) if n == v), None)

                if u_time is not None and v_time is not None:
                    edges_to_plot.append((u, v))
                    edge_colors.append((u_time + v_time) / 2)

        # Plot edges
        if edges_to_plot:
            for (u, v), color in zip(edges_to_plot, edge_colors):
                edge_xs = [G.nodes[u]['x'], G.nodes[v]['x']]
                edge_ys = [G.nodes[u]['y'], G.nodes[v]['y']]

                edge_color = cmap(color / max_time)

                ax.plot(edge_xs, edge_ys,
                       color=edge_color,
                       linewidth=0.5,
                       alpha=0.3,
                       zorder=2)

        # Plot nodes
        node_Xs = [G.nodes[node]['x'] for node in nodes]
        node_Ys = [G.nodes[node]['y'] for node in nodes]

        scatter = ax.scatter(node_Xs, node_Ys,
                           c=node_colors,
                           s=0.2,
                           cmap=cmap,
                           alpha=0.3,
                           vmin=0,
                           vmax=max_time,
                           zorder=3)

        # Plot garage location
        ax.scatter(result['garage_coords'][0],
                  result['garage_coords'][1],
                  c='black', s=2,
                  label=result['name'],
                  zorder=5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Travel time (minutes)', size=12)
    cbar.set_ticks([0, 15, 30, 45, 60])
    cbar.set_ticklabels(['0', '15', '30', '45', '60'])

    plt.title('Drive Times from RAC Approved Garages\n(1-hour radius)', pad=20)

    # Make the plot more focused on Britain
    plt.xlim(-8, 2)  # Longitude limits for Britain
    plt.ylim(50, 59)  # Latitude limits for Britain

    print(f"Saving plot to {filename}...")
    plt.savefig(filename, dpi=1000, bbox_inches='tight')
    plt.close()

def plot_garages_voronoi(G, garages, filename='uk_garages_voronoi.png'):
    """Plot garages and their Voronoi regions with UK map background"""
    # Create Voronoi regions
    vor = create_voronoi_regions(garages)

    # Create plot
    fig, ax = plt.subplots(figsize=(20, 20))

    # Set the coordinate system to Web Mercator for contextily
    ax.set_aspect('equal')

    # Plot base network (faint)
    ox.plot_graph(G, ax=ax, node_size=0, edge_color='#eeeeee',
                 edge_linewidth=0.3, show=False)

    # Add the UK map background
    ctx.add_basemap(ax,
                   crs='EPSG:4326',  # WGS84 coordinate system
                   source=ctx.providers.CartoDB.Positron,  # Light map style
                   alpha=0.5)  # Semi-transparent

    # Plot Voronoi regions
    for region in vor.regions:
        if -1 not in region and len(region) > 0:  # Valid region
            # Get vertices for this region
            polygon = vor.vertices[region]
            # Close the polygon by repeating first vertex
            polygon = np.vstack((polygon, polygon[0]))
            # Plot the polygon boundary
            ax.plot(polygon[:, 0], polygon[:, 1],
                   color='blue', linewidth=1,
                   alpha=0.7, linestyle='-',
                   zorder=2)

    # Plot garage locations
    for garage in garages['garageMarkers']:
        ax.scatter(float(garage['lng']), float(garage['lat']),
                  c='red', s=5, marker='o',
                  label=garage['companyName'],
                  zorder=3)

    plt.title('RAC Approved Garages and Service Areas\n(Voronoi Regions)', pad=20)

    # Make the plot more focused on Britain
    plt.xlim(-8, 2)  # Longitude limits for Britain
    plt.ylim(50, 59)  # Latitude limits for Britain

    print(f"Saving plot to {filename}...")
    plt.savefig(filename, dpi=1000, bbox_inches='tight')
    plt.close()

def main():
    print("Loading garage data...")
    with open('data/garages.json', 'r') as f:
        garages = json.load(f)

    print(f"Processing {len(garages['garageMarkers'])} garages")

    G = get_uk_network()

    # Plot both visualisations
    plot_garages_voronoi(G, garages)
    # create_isochrone_plot(G, garages, max_time=60, filename='uk_garage_isochrones_sample.png')

if __name__ == "__main__":
    main()

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shapely
import geovoronoi as gv
from shapely.geometry import Polygon, MultiPolygon
import osmnx as ox
import networkx as nx
from typing import Dict, Tuple, List
import os

# Constants
ROAD_SPEEDS = {
    'motorway': 70,  # mph
    'trunk': 50,
    'primary': 40,
    'secondary': 30,
    'tertiary': 30,
    'residential': 20,
    'unclassified': 20,
    'default': 30
}

def get_bristol_network(bristol_boundary):
    """Get road network for Bristol"""
    print("\nFetching Bristol road network...")
    try:
        G = ox.graph_from_polygon(bristol_boundary, network_type='drive')
        print(f"Downloaded network with {len(G.nodes)} nodes and {len(G.edges)} edges")

        print("Calculating travel times for road segments...")
        edge_count = 0
        for u, v, k, data in G.edges(data=True, keys=True):
            highway = data.get('highway', 'default')
            if isinstance(highway, list):
                highway = highway[0]
            speed = ROAD_SPEEDS.get(highway, ROAD_SPEEDS['default'])
            speed_ms = speed * 0.44704  # Convert mph to m/s

            if 'length' in data:
                data['travel_time'] = data['length'] / speed_ms
            else:
                if 'geometry' in data:
                    data['length'] = data['geometry'].length
                    data['travel_time'] = data['length'] / speed_ms
                else:
                    start = G.nodes[u]
                    end = G.nodes[v]
                    dist = ((start['x'] - end['x'])**2 + (start['y'] - end['y'])**2)**0.5
                    data['length'] = dist
                    data['travel_time'] = dist / speed_ms
            edge_count += 1
            if edge_count % 1000 == 0:
                print(f"Processed {edge_count} road segments...")

        print("Road network processing complete")
        return G
    except Exception as e:
        print(f"Error fetching network: {e}")
        return None

def generate_random_points(boundary: Polygon, n_points: int) -> np.ndarray:
    """Generate random points within boundary"""
    minx, miny, maxx, maxy = boundary.bounds
    n_samples = n_points * 4
    x = np.random.uniform(minx, maxx, n_samples)
    y = np.random.uniform(miny, maxy, n_samples)
    points = gpd.GeoSeries(gpd.points_from_xy(x, y))
    mask = points.within(boundary)
    return np.column_stack([x[mask], y[mask]])[:n_points]

def calculate_network_voronoi(G: nx.Graph, points: np.ndarray, boundary: Polygon, weight: str = 'travel_time') -> Dict[int, Polygon]:
    """Calculate network-based Voronoi regions using network distances and alpha shapes"""
    print(f"\nCalculating network-based Voronoi regions using '{weight}'...")

    # Find nearest nodes for each point
    print("Finding nearest network nodes for each point...")
    point_nodes = [ox.nearest_nodes(G, x, y) for x, y in points]
    print(f"Found {len(point_nodes)} nearest nodes")

    # Calculate shortest paths from each point to all nodes
    print("Calculating shortest paths from each point...")
    regions = {}
    for idx, start_node in enumerate(point_nodes):
        print(f"Processing point {idx + 1}/{len(point_nodes)}...")
        try:
            distances = nx.single_source_dijkstra_path_length(G, start_node, weight=weight)
            regions[idx] = distances
            print(f"Found paths to {len(distances)} nodes")
        except nx.NetworkXNoPath:
            print(f"Warning: No paths found for point {idx + 1}")
            continue

    # Assign nodes to closest point
    print("\nAssigning nodes to closest points...")
    node_assignments = {}
    node_count = len(G.nodes())
    processed = 0
    for node in G.nodes():
        min_time = float('inf')
        min_idx = None
        for idx, distances in regions.items():
            if node in distances and distances[node] < min_time:
                min_time = distances[node]
                min_idx = idx
        if min_idx is not None:
            node_assignments[node] = min_idx
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed}/{node_count} nodes...")

    # Create regions using network structure
    print("\nCreating network-based regions...")
    region_polygons = {}
    alpha = 0.0  # Start with convex hull
    min_points = 3  # Minimum points needed for a polygon

    for idx in range(len(points)):
        region_nodes = [node for node, assigned_idx in node_assignments.items() if assigned_idx == idx]
        if len(region_nodes) >= min_points:
            # Get node coordinates and connected edges
            node_coords = []
            edge_coords = []

            for node in region_nodes:
                node_coords.append((G.nodes[node]['x'], G.nodes[node]['y']))
                # Add points along connected edges
                for _, v, data in G.edges(node, data=True):
                    if node_assignments.get(v) == idx:  # Only if connected node is in same region
                        if 'geometry' in data:
                            # Sample points along the edge geometry
                            coords = list(data['geometry'].coords)
                            edge_coords.extend(coords)
                        else:
                            # Use straight line if no geometry
                            edge_coords.append((G.nodes[v]['x'], G.nodes[v]['y']))

            # Combine node and edge coordinates
            all_coords = node_coords + edge_coords
            if len(all_coords) >= min_points:
                # Create initial shape
                points_array = np.array(all_coords)
                try:
                    # Try creating alpha shape with increasing alpha until valid
                    while alpha <= 100:  # Limit the alpha value
                        try:
                            from alphashape import alphashape
                            region = alphashape(points_array, alpha)
                            if region.is_valid and region.area > 0:
                                final_poly = region.intersection(boundary)
                                if final_poly.is_valid and not final_poly.is_empty:
                                    region_polygons[idx] = final_poly
                                    print(f"Successfully created region {idx + 1} with alpha {alpha}")
                                    break
                        except Exception:
                            pass
                        alpha += 10

                    # Fallback to convex hull if alpha shape fails
                    if idx not in region_polygons:
                        hull = shapely.MultiPoint(all_coords).convex_hull
                        final_poly = hull.intersection(boundary)
                        if final_poly.is_valid and not final_poly.is_empty:
                            region_polygons[idx] = final_poly
                            print(f"Created region {idx + 1} using convex hull fallback")
                except Exception as e:
                    print(f"Error creating region {idx + 1}: {e}")

    print(f"Created {len(region_polygons)} valid regions")
    return region_polygons

def plot_region(region, ax, color, **kwargs):
    """Helper function to plot both Polygon and MultiPolygon geometries"""
    if isinstance(region, shapely.MultiPolygon):
        for poly in region.geoms:
            gpd.GeoDataFrame(geometry=[poly]).plot(ax=ax, color=color, **kwargs)
    elif isinstance(region, shapely.Polygon):
        gpd.GeoDataFrame(geometry=[region]).plot(ax=ax, color=color, **kwargs)

def main():
    print("Starting Bristol region Voronoi analysis...")

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Load and prepare Bristol and Wales region data
    print("\nLoading England and Wales WPC data...")
    try:
        england_wpc = gpd.read_file("data/England_WPC.json")
        wales_wpc = gpd.read_file("data/Wales_WPC.json")
        # Combine the datasets
        combined_wpc = pd.concat([england_wpc, wales_wpc], ignore_index=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure both England_WPC.json and Wales_WPC.json exist in the data directory")
        return

    print("Filtering for Bristol and South Wales region...")

    # Define the local authorities we want
    constituencies = [
        'Bristol West',
        'Bristol South',
        'Bristol East',
        'Bristol North West',
        'Filton and Bradley Stoke',
        'North Somerset',
        'Weston-Super-Mare',
        'North East Somerset',
        'Kingswood',
        'Thornbury and Yate',
        'Stroud',
        'Gloucester',
        'Forest of Dean',
        'Monmouth',
        'Newport East',
        'Newport West'
    ]

    # Filter for our region
    bristol_region = combined_wpc[combined_wpc['PCON13NM'].isin(constituencies)].copy()
    print(f"Found {len(bristol_region)} constituencies in the region")

    # Clean and create boundary
    print("\nCleaning and creating Bristol region boundary...")
    bristol_region.geometry = bristol_region.geometry.buffer(0)
    bristol_region = bristol_region[bristol_region.geometry.is_valid]
    boundary_shape = shapely.ops.unary_union(bristol_region.geometry)

    # Don't simplify to just the exterior - keep the full shape
    if isinstance(boundary_shape, MultiPolygon):
        # Instead of taking just the largest polygon, merge all polygons
        boundary_shape = shapely.ops.unary_union(boundary_shape)

    print("Bristol region boundary created successfully")

    # Load garage coordinates and filter for Bristol region
    print("\nLoading and filtering garage locations...")
    garages_df = pd.read_csv("data/garage_coords.csv")
    # Convert to GeoDataFrame
    garages_gdf = gpd.GeoDataFrame(
        garages_df,
        geometry=gpd.points_from_xy(garages_df.Longitude, garages_df.Latitude)
    )
    # Filter for garages within the boundary
    bristol_garages = garages_gdf[garages_gdf.geometry.within(boundary_shape)]
    print(f"Found {len(bristol_garages)} garages in the Bristol region")

    # Extract coordinates for Voronoi calculation
    points = np.column_stack([
        bristol_garages.geometry.x,
        bristol_garages.geometry.y
    ])

    if len(points) == 0:
        print("Error: No garages found in the Bristol region!")
        return

    print(f"Using {len(points)} garage locations for Voronoi analysis")

    # Create separate figures
    print("\nCreating visualizations...")

    # Figure 1: Euclidean Voronoi
    print("\nCreating Euclidean Voronoi plot...")
    fig1, ax1 = plt.subplots(figsize=(15, 15))

    print("\nCalculating Euclidean Voronoi regions...")
    poly_shapes, pts = gv.voronoi_regions_from_coords(points, boundary_shape)
    print(f"Created {len(poly_shapes)} Euclidean Voronoi regions")

    print("Plotting Euclidean Voronoi diagram...")
    for idx, region in enumerate(poly_shapes.values()):
        gpd.GeoDataFrame(geometry=[region]).plot(
            ax=ax1,
            color=plt.cm.Set3(idx / len(points)),
            alpha=0.5,
            edgecolor='black'
        )
    gpd.GeoDataFrame(geometry=gpd.points_from_xy(points[:, 0], points[:, 1])).plot(
        ax=ax1, color='red', markersize=20
    )
    ax1.set_title('Greater Bristol Region - Euclidean Voronoi')

    # Add boundary and format
    gpd.GeoDataFrame(geometry=[boundary_shape.boundary]).plot(
        ax=ax1, color='black', linewidth=1
    )
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.tight_layout()
    print("Saving Euclidean Voronoi plot...")
    plt.savefig('plots/euclidean_voronoi.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Network and Traffic-Weighted Voronoi
    print("\nCreating Network and Traffic-Weighted Voronoi plots...")
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(30, 15))

    # 2. Network Voronoi
    print("\nProcessing Network Voronoi...")
    G = get_bristol_network(boundary_shape)
    if G is not None:
        # Plot road network first (only once) - more efficiently
        print("Plotting Bristol road network...")
        # Collect all geometries into a list first
        road_geometries = []
        for _, _, data in G.edges(data=True):
            if 'geometry' in data:
                road_geometries.append(data['geometry'])

        # Create single GeoDataFrame for all roads
        roads_gdf = gpd.GeoDataFrame(geometry=road_geometries)

        # Plot once for each axis
        roads_gdf.plot(ax=ax2, color='black', linewidth=0.5, alpha=0.3)
        roads_gdf.plot(ax=ax3, color='black', linewidth=0.5, alpha=0.3)

        # Calculate and plot Network Voronoi regions
        network_regions = calculate_network_voronoi(G, points, boundary_shape)
        print("Plotting Network Voronoi regions...")
        for idx, region in network_regions.items():
            plot_region(
                region,
                ax2,
                color=plt.cm.Set3(idx / len(points)),
                alpha=0.5,
                edgecolor='black'
            )

        # Plot points
        gpd.GeoDataFrame(geometry=gpd.points_from_xy(points[:, 0], points[:, 1])).plot(
            ax=ax2, color='red', markersize=20
        )
        ax2.set_title('Greater Bristol Region - Network Voronoi (Travel Time)')

        # 3. Traffic-Weighted Voronoi
        print("\nProcessing Traffic-Weighted Voronoi...")
        print("Applying traffic weights to road network...")
        TRAFFIC_FACTORS = {
            'motorway': 2.5,  # Higher congestion on major roads
            'trunk': 2.0,
            'primary': 1.8,
            'secondary': 1.2,
            'tertiary': 1.1,
            'residential': 1.0,
            'unclassified': 1.0,
            'default': 1.0
        }

        edge_count = 0
        for u, v, k, data in G.edges(data=True, keys=True):
            highway = data.get('highway', 'default')
            if isinstance(highway, list):
                highway = highway[0]
            traffic_factor = TRAFFIC_FACTORS.get(highway, TRAFFIC_FACTORS['default'])
            data['traffic_time'] = data['travel_time'] * traffic_factor
            edge_count += 1
            if edge_count % 1000 == 0:
                print(f"Applied traffic weights to {edge_count} edges...")

        print("Calculating traffic-weighted regions...")
        traffic_regions = calculate_network_voronoi(G, points, boundary_shape, weight='traffic_time')
        print("Plotting Traffic-Weighted Voronoi regions...")
        for idx, region in traffic_regions.items():
            plot_region(
                region,
                ax3,
                color=plt.cm.Set3(idx / len(points)),
                alpha=0.5,
                edgecolor='black'
            )

        # Plot points (already plotted road network earlier)
        gpd.GeoDataFrame(geometry=gpd.points_from_xy(points[:, 0], points[:, 1])).plot(
            ax=ax3, color='red', markersize=20
        )
        ax3.set_title('Greater Bristol Region - Traffic-Weighted Voronoi')

        # Add Bristol boundary to network plots and fix aspect ratio
        print("\nAdding Bristol boundary to network plots...")
        for ax in [ax2, ax3]:
            gpd.GeoDataFrame(geometry=[boundary_shape.boundary]).plot(
                ax=ax, color='black', linewidth=1
            )
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([])
            ax.set_yticks([])

        print("\nFinalizing visualization...")
        plt.tight_layout()
        print("Saving Network and Traffic-Weighted Voronoi plots...")
        plt.savefig('plots/network_voronoi.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
import json
import os
import pickle
import numpy as np
import folium
from folium import plugins
from shapely import geometry
from shapely.ops import unary_union
from shapely.prepared import prep
import osmnx as ox
import matplotlib.pyplot as plt
from geovoronoi import voronoi_regions_from_coords
import contextily as ctx
import concurrent.futures
import networkx as nx
from pytopojson import topology, feature
import pandas as pd
from scipy.spatial import Voronoi
import hashlib
import shapely

# Add type hints and organize imports better
from typing import List, Tuple, Dict, Any
from pathlib import Path

# Constants at the top of the file
CACHE_DIR = Path("data")
UK_CENTER = (54.7, -3.5)  # Approximate centre of UK
ROAD_SPEEDS = {
    'motorway': 60,  # mph
    'trunk': 50,
    'primary': 40,
    'secondary': 30,
    'tertiary': 30,
    'residential': 20,
    'unclassified': 20,
    'default': 30
}

# Helper functions first
def ensure_cache_dir() -> None:
    """Ensure the cache directory exists"""
    CACHE_DIR.mkdir(exist_ok=True)

def process_geojson_features(features: List[Dict]) -> List[geometry.base.BaseGeometry]:
    """Process GeoJSON features into Shapely geometries"""
    geometries = []
    for feature_obj in features:
        geom = feature_obj['geometry']
        try:
            if geom['type'] == 'Polygon':
                coords = geom['coordinates']
                exterior = coords[0]
                interiors = coords[1:] if len(coords) > 1 else []
                poly = geometry.Polygon(exterior, interiors)
                if poly.is_valid:
                    geometries.append(poly)
                else:
                    fixed_poly = poly.buffer(0)
                    if fixed_poly.is_valid:
                        geometries.append(fixed_poly)
            elif geom['type'] == 'MultiPolygon':
                multi_poly = geometry.MultiPolygon([
                    geometry.Polygon(poly[0], poly[1:] if len(poly) > 1 else [])
                    for poly in geom['coordinates']
                ])
                if multi_poly.is_valid:
                    geometries.append(multi_poly)
                else:
                    fixed_poly = multi_poly.buffer(0)
                    if fixed_poly.is_valid:
                        geometries.append(fixed_poly)
        except Exception as e:
            print(f"Warning: Skipping invalid geometry: {e}")
            continue
    return geometries

# Core data loading functions
def get_or_create_network():
    """Load network from cache if it exists, otherwise create and cache it"""
    cache_filename = CACHE_DIR / "uk_network.graphml"

    if cache_filename.exists():
        print("Loading cached street network...")
        return ox.load_graphml(cache_filename)

    print("Downloading and processing street network...")
    cf = '["highway"~"motorway|trunk|primary|secondary"]'
    G = ox.graph_from_place("United Kingdom",
                           network_type="drive",
                           custom_filter=cf,
                           simplify=True)

    ensure_cache_dir()
    print("Caching street network...")
    ox.save_graphml(G, str(cache_filename))

    return G

def get_active_garages(csv_path: str = 'data/garage_coords.csv') -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """Get list of garages from CSV file with consistent indexing"""
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert to the format we need
    active_garages = []
    active_garage_data = []

    for _, row in df.iterrows():
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            active_garages.append([float(row['Longitude']), float(row['Latitude'])])
            active_garage_data.append({
                'companyName': row['Postcode'],
                'lat': float(row['Latitude']),
                'lng': float(row['Longitude'])
            })

    return active_garages, active_garage_data

def get_uk_boundary() -> Tuple[geometry.base.BaseGeometry, List[geometry.base.BaseGeometry]]:
    """Get UK boundary polygons from Westminster Parliamentary Constituencies TopoJSON files.
    Returns both the merged outer boundary and individual constituency boundaries.
    Uses cached version if available.
    """
    cache_file = CACHE_DIR / 'uk_boundaries.pkl'

    # Try to load from cache first
    try:
        print("Loading UK boundaries from cache...")
        with open(cache_file, 'rb') as f:
            boundaries = pickle.load(f)
            outer_boundary = boundaries['outer']
            constituencies = boundaries['constituencies']

            # Validate the geometries
            if not isinstance(outer_boundary, (geometry.Polygon, geometry.MultiPolygon)):
                raise ValueError("Cached outer boundary is not a Polygon or MultiPolygon")
            if not all(isinstance(p, (geometry.Polygon, geometry.MultiPolygon)) for p in constituencies):
                raise ValueError("Cached constituencies contain invalid geometries")

            return outer_boundary, constituencies
    except (FileNotFoundError, pickle.UnpicklingError, ValueError) as e:
        print(f"Cache not found or invalid ({e}), processing TopoJSON...")

        # Process TopoJSON files for all countries
        constituencies = []

        # Process England
        print("Processing England constituencies...")
        with open('data/England_WPC.json') as f:
            england_topo = json.load(f)
            feature_converter = feature.Feature()
            england_features = feature_converter(england_topo, england_topo['objects']['wpc'])['features']
            constituencies.extend(process_geojson_features(england_features))

        # Process Wales
        print("Processing Wales constituencies...")
        with open('data/Wales_WPC.json') as f:
            wales_topo = json.load(f)
            feature_converter = feature.Feature()
            wales_features = feature_converter(wales_topo, wales_topo['objects']['wpc'])['features']
            constituencies.extend(process_geojson_features(wales_features))

        # Process Scotland
        print("Processing Scotland constituencies...")
        with open('data/Scotland_WPC.json') as f:
            scotland_topo = json.load(f)
            feature_converter = feature.Feature()
            scotland_features = feature_converter(scotland_topo, scotland_topo['objects']['lad'])['features']
            constituencies.extend(process_geojson_features(scotland_features))

        # Process Northern Ireland
        print("Processing Northern Ireland constituencies...")
        with open('data/NI_WPC.json') as f:
            ni_topo = json.load(f)
            feature_converter = feature.Feature()
            ni_features = feature_converter(ni_topo, ni_topo['objects']['wpc'])['features']
            constituencies.extend(process_geojson_features(ni_features))

        if not constituencies:
            raise ValueError("No valid polygons found in the TopoJSON data")

        print(f"Processing {len(constituencies)} constituencies...")

        # Create the outer boundary by merging all constituencies
        outer_boundary = unary_union(constituencies)

        # Ensure geometries are valid
        if not outer_boundary.is_valid:
            print("Fixing outer boundary geometry...")
            outer_boundary = outer_boundary.buffer(0)

        # Final validity checks
        if not outer_boundary.is_valid:
            print("Final outer boundary fix...")
            outer_boundary = outer_boundary.buffer(0)

        constituencies = [p.buffer(0) if not p.is_valid else p for p in constituencies]

        # Save to cache
        try:
            print("Saving UK boundaries to cache...")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'outer': outer_boundary,
                    'constituencies': constituencies
                }, f)
            print("Cache saved successfully")
        except Exception as e:
            print(f"Warning: Could not save boundary cache: {e}")

        return outer_boundary, constituencies

def get_voronoi_regions(uk_garages: np.ndarray, uk_boundary: geometry.base.BaseGeometry) -> Tuple[Dict[int, geometry.Polygon], Dict[int, List[int]]]:
    """Get Voronoi regions from cache or generate new"""
    cache_file = CACHE_DIR / 'voronoi_cache.pkl'

    # Create stable hash key
    hasher = hashlib.sha256()

    # Hash garage coordinates (sorted for order invariance)
    sorted_garages = sorted(uk_garages.tolist())
    hasher.update(repr(sorted_garages).encode('utf-8'))

    # Ensure boundary is valid and closed
    if isinstance(uk_boundary, geometry.MultiPolygon):
        # Convert MultiPolygon to a single Polygon by taking the union
        normalized_boundary = shapely.ops.unary_union([poly.buffer(0) for poly in uk_boundary.geoms])
    else:
        normalized_boundary = uk_boundary.buffer(0)

    # Additional validation and fixing steps
    if not normalized_boundary.is_valid:
        print("Fixing invalid boundary...")
        normalized_boundary = normalized_boundary.buffer(0)

    if not normalized_boundary.is_simple:
        print("Simplifying complex boundary...")
        normalized_boundary = normalized_boundary.simplify(0.0001)

    # Ensure the boundary is closed
    if isinstance(normalized_boundary, geometry.Polygon):
        if not normalized_boundary.exterior.is_ring:
            print("Closing boundary ring...")
            coords = list(normalized_boundary.exterior.coords)
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            normalized_boundary = geometry.Polygon(coords)

    # Hash the normalized boundary
    hasher.update(normalized_boundary.wkb)
    hash_key = hasher.hexdigest()

    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            if cached_data['hash'] == hash_key:
                print("Using cached Voronoi regions")
                return cached_data['regions'], cached_data['points']
    except (FileNotFoundError, KeyError, pickle.UnpicklingError):
        pass

    # Cache miss - generate new
    print("Generating new Voronoi regions...")
    regions, points = voronoi_regions_from_coords(uk_garages, normalized_boundary)

    # Validate the generated regions
    for region_id, poly in regions.items():
        if isinstance(poly, geometry.MultiPolygon):
            # Ensure each part is within the boundary
            valid_parts = [p for p in poly.geoms if p.intersection(normalized_boundary).area > 0]
            if valid_parts:
                regions[region_id] = geometry.MultiPolygon(valid_parts)
        else:
            # Ensure the polygon is within the boundary
            regions[region_id] = poly.intersection(normalized_boundary)

    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'hash': hash_key,
            'regions': regions,
            'points': points
        }, f)

    return regions, points

def create_interactive_map(uk_boundary: geometry.base.BaseGeometry, constituencies: List[geometry.base.BaseGeometry], filename: str = 'interactive_garages_map.html') -> None:
    """Create an interactive map with garage locations and Voronoi tessellation"""
    m = folium.Map(location=UK_CENTER,
                   zoom_start=6,
                   tiles='cartodbpositron',
                   prefer_canvas=True)

    # Create feature groups for different layers
    marker_cluster = plugins.MarkerCluster()
    voronoi_layer = folium.FeatureGroup(name='Voronoi Regions')
    constituency_layer = folium.FeatureGroup(name='Constituencies')
    outer_boundary_layer = folium.FeatureGroup(name='UK Boundary')

    # Add UK boundary to its layer
    if isinstance(uk_boundary, geometry.MultiPolygon):
        for poly in uk_boundary.geoms:
            coords = [[y, x] for x, y in poly.exterior.coords]
            folium.Polygon(
                locations=coords,
                color='black',
                weight=2,
                fill=False,
                opacity=1.0,
                smoothFactor=0.0
            ).add_to(outer_boundary_layer)
    else:
        coords = [[y, x] for x, y in uk_boundary.exterior.coords]
        folium.Polygon(
            locations=coords,
            color='black',
            weight=2,
            fill=False,
            opacity=1.0,
            smoothFactor=0.0
        ).add_to(outer_boundary_layer)

    # Get active garages with consistent indexing
    active_garages, active_garage_data = get_active_garages()
    print(f"Loaded {len(active_garages)} garages")

    # Filter garages within UK boundary
    prepared_boundary = prep(uk_boundary.buffer(0.001))
    points = [geometry.Point(p) for p in active_garages]
    mask = [prepared_boundary.contains(p) for p in points]
    valid_indices = [i for i, valid in enumerate(mask) if valid]

    uk_garages = np.array([g for g, m in zip(active_garages, mask) if m])
    uk_garage_data = [d for d, m in zip(active_garage_data, mask) if m]

    # Create Voronoi regions for valid garages
    print("Creating Voronoi regions...")
    region_polys, region_pts = get_voronoi_regions(uk_garages, uk_boundary)

    # Add Voronoi regions to map
    for region_id, poly in region_polys.items():
        if isinstance(poly, geometry.MultiPolygon):
            for part in poly.geoms:
                coords = [[y, x] for x, y in part.exterior.coords]
                folium.Polygon(
                    locations=coords,
                    color='blue',
                    weight=1,
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.1,
                    smoothFactor=0.0
                ).add_to(voronoi_layer)
        else:
            coords = [[y, x] for x, y in poly.exterior.coords]
            folium.Polygon(
                locations=coords,
                color='blue',
                weight=1,
                fill=True,
                fill_color='blue',
                fill_opacity=0.1,
                smoothFactor=0.0
            ).add_to(voronoi_layer)

    # Add markers for all garages with correct indices
    for idx, garage in enumerate(uk_garage_data):
        original_index = valid_indices[idx]
        popup_content = f"""
            <b>{garage['companyName']}</b><br>
            Filtered Index: {idx}<br>
            Original Index: {original_index}<br>
            Lat: {garage['lat']:.4f}<br>
            Lng: {garage['lng']:.4f}
        """

        folium.Marker(
            location=[float(garage['lat']), float(garage['lng'])],
            popup=popup_content,
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(marker_cluster)

    for constituency in constituencies:
        if isinstance(constituency, geometry.MultiPolygon):
            for part in constituency.geoms:
                coords = [[y, x] for x, y in part.exterior.coords]
                folium.Polygon(
                    locations=coords,
                    color='black',
                    weight=1,
                    fill=False,
                    smoothFactor=0.0
                ).add_to(constituency_layer)
        else:
            coords = [[y, x] for x, y in constituency.exterior.coords]
            folium.Polygon(
                locations=coords,
                color='black',
                weight=1,
                fill=False,
                smoothFactor=0.0
            ).add_to(constituency_layer)

    # Add all layers to the map
    marker_cluster.add_to(m)
    voronoi_layer.add_to(m)
    constituency_layer.add_to(m)
    outer_boundary_layer.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add fullscreen button
    plugins.Fullscreen().add_to(m)

    # Save the map with high DPI
    m.save(filename)

def fetch_network_and_isochrones(
    selected_poly: geometry.base.BaseGeometry,
    selected_garage: Dict[str, Any],
    uk_boundary: geometry.base.BaseGeometry,
    buffer_size: float,  # In kilometers
    speeds: Dict[str, int]
) -> Tuple[nx.Graph, Dict[int, geometry.base.BaseGeometry]]:
    """Fetch road network and calculate isochrones for a garage"""
    cache_dir = CACHE_DIR / 'road_networks'
    cache_dir.mkdir(exist_ok=True)

    # Generate cache filename based on garage coordinates
    garage_lat = selected_garage['lat']
    garage_lng = selected_garage['lng']
    cache_file = cache_dir / f'network_{garage_lat:.4f}_{garage_lng:.4f}.graphml'

    # Convert km buffer to approximate degrees (1km ≈ 0.009°)
    buffer_deg = buffer_size * 0.009
    garage_point = geometry.Point(garage_lng, garage_lat)

    # Try to load from cache first
    G = None
    try:
        if cache_file.exists():
            print(f"Loading cached road network for {garage_lat:.4f}, {garage_lng:.4f}")
            G = ox.load_graphml(cache_file)
    except Exception as e:
        print(f"Cache load failed: {e}")
        G = None

    # If cache miss, download fresh data
    if G is None:
        print(f"Downloading road network for {garage_lat:.4f}, {garage_lng:.4f}...")
        try:
            if isinstance(selected_poly, geometry.MultiPolygon):
                # Download network for each part separately
                graphs = []
                for i, poly in enumerate(selected_poly.geoms):
                    # Create search area for this part
                    search_area = poly.buffer(buffer_deg)
                    if poly.contains(garage_point):
                        # Add extra buffer around garage location for its polygon
                        search_area = search_area.union(garage_point.buffer(buffer_deg * 2))

                    # Clip to UK boundary
                    search_area = search_area.intersection(uk_boundary)
                    if not search_area.is_empty:
                        try:
                            g = ox.graph_from_polygon(search_area, network_type='drive')
                            if g is not None and len(g.nodes) > 0:
                                graphs.append(g)
                        except Exception as e:
                            if "No data elements" in str(e):
                                print(f"Info: No road data found in region part {i+1}")
                            else:
                                print(f"Warning: Failed to download network for part {i+1}: {e}")

                # Combine all graphs
                if graphs:
                    G = nx.compose_all(graphs)
                else:
                    print("No road data found in any region part")
                    return nx.Graph(), {}
            else:
                # Single polygon case
                search_area = selected_poly.buffer(buffer_deg).union(
                    garage_point.buffer(buffer_deg * 2)
                )
                search_area = search_area.intersection(uk_boundary)
                if search_area.is_empty:
                    return nx.Graph(), {}
                G = ox.graph_from_polygon(search_area, network_type='drive')

            # Cache the combined network
            ox.save_graphml(G, cache_file)
        except Exception as e:
            print(f"Failed to download network: {e}")
            return nx.Graph(), {}

    # Add travel time to edges based on length and speed limit
    for u, v, k, data in G.edges(data=True, keys=True):
        # Get the highway type and corresponding speed
        highway = data.get('highway', 'default')
        if isinstance(highway, list):
            highway = highway[0]
        speed = speeds.get(highway, speeds['default'])

        # Convert speed to meters per second (1 mph = 0.44704 m/s)
        speed_ms = speed * 0.44704

        # Calculate travel time in seconds
        if 'length' in data:
            data['travel_time'] = data['length'] / speed_ms
        else:
            # If length is missing, estimate it from the geometry
            if 'geometry' in data:
                data['length'] = data['geometry'].length
                data['travel_time'] = data['length'] / speed_ms
            else:
                # If no geometry, use straight-line distance
                start = G.nodes[u]
                end = G.nodes[v]
                dist = ((start['x'] - end['x'])**2 + (start['y'] - end['y'])**2)**0.5
                data['length'] = dist
                data['travel_time'] = dist / speed_ms

    # Get garage coordinates
    garage_lat = float(selected_garage['lat'])
    garage_lng = float(selected_garage['lng'])

    # Find the nearest node to the garage
    garage_node = ox.nearest_nodes(G, garage_lng, garage_lat)

    # Calculate shortest paths based on travel time
    time_distances = {}
    for node in G.nodes():
        try:
            time_distances[node] = nx.shortest_path_length(G, garage_node, node, weight='travel_time')
        except nx.NetworkXNoPath:
            continue

    # Create isochrone polygons
    isochrone_polys = {}
    for minutes in [30, 10]:  # 30 minutes first, then 10 minutes
        # Convert minutes to seconds
        max_time = minutes * 60

        # Get nodes within time limit
        nodes_within_time = [node for node, time in time_distances.items() if time <= max_time]

        if nodes_within_time:
            # Get coordinates for these nodes
            coords = [[G.nodes[node]['x'], G.nodes[node]['y']] for node in nodes_within_time]

            # Create a polygon from the convex hull of these points
            points = [geometry.Point(x, y) for x, y in coords]
            points_poly = geometry.MultiPoint(points).convex_hull

            # Buffer the polygon slightly and clip to the Voronoi region
            buffered = points_poly.buffer(0.001)
            clipped = buffered.intersection(selected_poly)

            isochrone_polys[minutes] = clipped

    return G, isochrone_polys

def process_garage_data(args):
    """Helper function for parallel processing of garage data"""
    return fetch_network_and_isochrones(*args)

def get_voronoi_mapping(active_garages: List[List[float]], uk_boundary: geometry.base.BaseGeometry) -> Tuple[Dict[int, int], Dict[int, geometry.Polygon], List[int]]:
    """Create mapping between original garage indices and Voronoi regions"""
    # Filter garages within UK boundary
    prepared_boundary = prep(uk_boundary.buffer(0.001))
    mask = [prepared_boundary.contains(geometry.Point(p)) for p in active_garages]
    valid_indices = [i for i, valid in enumerate(mask) if valid]  # Original CSV indices of valid garages

    # Generate Voronoi regions for valid garages
    uk_garages = np.array([g for g, m in zip(active_garages, mask) if m])
    region_polys, region_pts = voronoi_regions_from_coords(uk_garages, uk_boundary)

    # Create mapping {filtered_index: region_id}
    index_map = {}
    for region_id, pt_indices in region_pts.items():
        # pt_indices[0] = index in uk_garages (filtered list)
        filtered_index = pt_indices[0]
        index_map[filtered_index] = region_id

    return index_map, region_polys, valid_indices

def plot_garage_region(filtered_index: int, uk_boundary: geometry.base.BaseGeometry, output_dir: Path = Path("plots")) -> None:
    """Plot a single garage's Voronoi region with road network"""
    # Load data
    active_garages, active_garage_data = get_active_garages()
    print(f"Loaded {len(active_garages)} garages")

    print(f"Getting Voronoi mapping...")
    # Get mapping and validate
    index_map, region_polys, valid_indices = get_voronoi_mapping(active_garages, uk_boundary)
    if filtered_index < 0 or filtered_index >= len(valid_indices):
        raise ValueError(f"Invalid filtered index: {filtered_index}")

    original_index = valid_indices[filtered_index]
    garage = active_garage_data[original_index]
    region = region_polys[index_map[filtered_index]]
    print(f"Got Voronoi mapping for garage")

    # Validate spatial alignment
    print(f"Validating spatial alignment...")
    garage_point = geometry.Point(garage['lng'], garage['lat'])
    if not region.contains(garage_point):
        print(f"Warning: Garage {filtered_index} is not within its Voronoi region!")
    print(f"Validated spatial alignment")

    # Fetch road network with 1km buffer
    print("Fetching road network...")
    G, _ = fetch_network_and_isochrones(
        region,
        garage,
        uk_boundary,
        buffer_size=0,
        speeds=ROAD_SPEEDS
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot Voronoi region
    if isinstance(region, geometry.MultiPolygon):
        for poly in region.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='blue', alpha=1.0, lw=1.0, linestyle='-', zorder=0)
    else:
        x, y = region.exterior.xy
        ax.plot(x, y, color='blue', alpha=1.0, lw=1.0, linestyle='-', zorder=0)

    # Plot roads
    for u, v, data in G.edges(data=True):
        if 'geometry' in data:
            xs, ys = data['geometry'].xy
            ax.plot(xs, ys, color='black', lw=0.5, alpha=1.0, zorder=1)
        else:
            x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
            x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
            ax.plot([x1, x2], [y1, y2], color='black', lw=0.5, alpha=1.0, zorder=1)

    # Plot garage
    ax.scatter(garage['lng'], garage['lat'], c='red', s=100,
               edgecolor='white', zorder=2, label='Garage')

    # Final styling
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Coverage Region: {garage['companyName']}")
    ax.legend()
    plt.savefig(output_dir / f"garage_{filtered_index}_coverage.png", dpi=300, bbox_inches='tight')
    plt.close()

def main() -> None:
    """Main entry point for the garage mapping application"""
    print("Starting garage mapping...")

    # Load UK boundaries once
    print("Loading UK boundaries...")
    uk_boundary, constituencies = get_uk_boundary()

    # Create interactive map
    print("Creating interactive visualisation...")
    create_interactive_map(uk_boundary, constituencies)
    print("Interactive map created")

    try:
        print("\nPlotting example garage region...")
        plot_garage_region(434, uk_boundary)
        print("Example plot saved to plots/")
    except ValueError as e:
        print(f"Error plotting example: {e}")

    print("All operations completed")

if __name__ == "__main__":
    main()
import json
import os
import pickle
import numpy as np
import folium
from folium import plugins
from shapely import geometry
from shapely.ops import unary_union
import osmnx as ox
import matplotlib.pyplot as plt
from geovoronoi import voronoi_regions_from_coords
import contextily as ctx
import concurrent.futures
import networkx as nx
from pytopojson import topology, feature
import pandas as pd
from scipy.spatial import Voronoi
from shapely.prepared import prep

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
                'companyName': row['Postcode'],  # Use postcode as the name
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

# Visualization functions
def create_interactive_map(filename: str = 'interactive_garages_map.html') -> None:
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

    # Get active garages with consistent indexing
    active_garages, active_garage_data = get_active_garages()

    # Get UK boundaries
    outer_boundary, constituencies = get_uk_boundary()

    # Quick validation using prepared geometry
    prepared_boundary = prep(outer_boundary.buffer(0.001))  # Small buffer for numerical precision
    points = [geometry.Point(p) for p in active_garages]
    mask = [prepared_boundary.contains(p) for p in points]

    uk_garages = np.array([g for g, m in zip(active_garages, mask) if m])
    uk_garage_data = [d for d, m in zip(active_garage_data, mask) if m]

    # Create Voronoi regions for valid garages
    print("Creating Voronoi regions...")
    region_polys, region_pts = voronoi_regions_from_coords(uk_garages, outer_boundary)

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

    # Add markers for all garages
    for idx, garage in enumerate(uk_garage_data):
        popup_content = f"""
            <b>{garage['companyName']}</b><br>
            Plot Index: {idx}<br>
            Lat: {garage['lat']}<br>
            Lng: {garage['lng']}
        """

        folium.Marker(
            location=[float(garage['lat']), float(garage['lng'])],
            popup=popup_content,
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(marker_cluster)

    # Add constituency boundaries
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
    buffer_size: float,
    speeds: Dict[str, int]
) -> Tuple[nx.Graph, Dict[int, geometry.base.BaseGeometry]]:
    """Fetch road network and calculate isochrones for a garage"""
    cache_dir = CACHE_DIR / 'road_networks'
    cache_dir.mkdir(exist_ok=True)

    # Generate cache filename based on garage ID
    garage_id = selected_garage.get('id', 'unknown')
    cache_file = cache_dir / f'network_{garage_id}.graphml'

    # Buffer the Voronoi polygon slightly to ensure we get all relevant roads
    buffered_poly = selected_poly.buffer(buffer_size)

    # Clip to UK boundary
    buffered_poly = buffered_poly.intersection(uk_boundary)

    # Try to load from cache first
    G = None
    try:
        if os.path.exists(cache_file):
            print(f"Loading cached road network for garage {garage_id}...")
            G = ox.load_graphml(cache_file)
    except Exception as e:
        print(f"Cache load failed for garage {garage_id}: {e}")
        G = None

    # If not in cache or cache load failed, download and cache
    if G is None:
        print(f"Downloading road network for garage {garage_id}...")
        G = ox.graph_from_polygon(buffered_poly, network_type='drive')

        # Save to cache
        try:
            print(f"Caching road network for garage {garage_id}...")
            ox.save_graphml(G, cache_file)
        except Exception as e:
            print(f"Failed to cache network for garage {garage_id}: {e}")

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

def main() -> None:
    """Main entry point for the garage mapping application"""
    print("Starting garage mapping...")

    active_garages, active_garage_data = get_active_garages()
    print(f"Loaded {len(active_garages)} garages from CSV")

    print("Creating interactive visualisation...")
    create_interactive_map()
    print("Map saved as 'interactive_garages_map.html'")

if __name__ == "__main__":
    main()
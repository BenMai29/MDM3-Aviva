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
import geopandas as gpd
from tqdm import tqdm

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

def ensure_cache_dir() -> None:
    """Ensure the cache directory exists"""
    CACHE_DIR.mkdir(exist_ok=True)

def process_geojson_features(features: List[Dict]) -> List[geometry.base.BaseGeometry]:
    """Process GeoJSON features into Shapely geometries"""
    geometries = []
    for feature_obj in tqdm(features, desc="Processing GeoJSON features"):
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
    df = pd.read_csv(csv_path)
    active_garages = []
    active_garage_data = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Loading active garages"):
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
    try:
        print("Loading UK boundaries from cache...")
        with open(cache_file, 'rb') as f:
            boundaries = pickle.load(f)
            outer_boundary = boundaries['outer']
            constituencies = boundaries['constituencies']
            return outer_boundary, constituencies
    except (FileNotFoundError, pickle.UnpicklingError, ValueError) as e:
        print(f"Cache not found or invalid ({e}), processing TopoJSON...")
        constituencies = []
        with open('data/England_WPC.json') as f:
            england_topo = json.load(f)
            feature_converter = feature.Feature()
            england_features = feature_converter(england_topo, england_topo['objects']['wpc'])['features']
            constituencies.extend(process_geojson_features(england_features))
        outer_boundary = unary_union(constituencies)
        outer_boundary = outer_boundary.buffer(0) if not outer_boundary.is_valid else outer_boundary
        constituencies = [p.buffer(0) if not p.is_valid else p for p in constituencies]
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'outer': outer_boundary, 'constituencies': constituencies}, f)
        except Exception as e:
            print(f"Warning: Could not save boundary cache: {e}")
        return outer_boundary, constituencies

def create_interactive_map(filename: str = 'interactive_garages_map.html', road_graph=None, voronoi_traffic_data=None) -> None:
    """Create an interactive map with garage locations, Voronoi tessellation, and traffic intensity"""
    m = folium.Map(location=UK_CENTER, zoom_start=6, tiles='cartodbpositron', prefer_canvas=True)
    marker_cluster = plugins.MarkerCluster()
    voronoi_layer = folium.FeatureGroup(name='Voronoi Regions')
    constituency_layer = folium.FeatureGroup(name='Constituencies')
    outer_boundary_layer = folium.FeatureGroup(name='UK Boundary')
    traffic_layer = folium.FeatureGroup(name='Traffic Intensity')

    # Traffic color mapping function
    def traffic_color(intensity: float) -> str:
        """Map traffic intensity to an intuitive color."""
        if intensity < 0.33:
            return 'green'  # Low traffic
        elif intensity < 0.66:
            return 'yellow'  # Medium traffic
        else:
            return 'red'  # High traffic

    active_garages, active_garage_data = get_active_garages()
    outer_boundary, constituencies = get_uk_boundary()
    prepared_boundary = prep(outer_boundary.buffer(0.001))
    points = [geometry.Point(p) for p in active_garages]
    mask = [prepared_boundary.contains(p) for p in points]
    uk_garages = np.array([g for g, m in zip(active_garages, mask) if m])
    uk_garage_data = [d for d, m in zip(active_garage_data, mask) if m]

    print("Creating Voronoi regions...")
    region_polys, region_pts = voronoi_regions_from_coords(uk_garages, outer_boundary)
    for region_id, poly in tqdm(region_polys.items(), desc="Adding Voronoi regions"):
        traffic_intensity = voronoi_traffic_data.get(region_id, 0) if voronoi_traffic_data else 0
        color_hex = traffic_color(traffic_intensity / 2)  # Normalize intensity and map to color

        if isinstance(poly, geometry.MultiPolygon):
            for sub_poly in poly.geoms:
                coords = [[y, x] for x, y in sub_poly.exterior.coords]
                folium.Polygon(locations=coords, color='black', weight=1, fill=True, fill_color=color_hex, fill_opacity=0.6).add_to(voronoi_layer)
        else:
            coords = [[y, x] for x, y in poly.exterior.coords]
            folium.Polygon(locations=coords, color='black', weight=1, fill=True, fill_color=color_hex, fill_opacity=0.6).add_to(voronoi_layer)

    for idx, garage in enumerate(tqdm(uk_garage_data, desc="Adding garage markers")):
        popup_content = f"""<b>{garage['companyName']}</b><br>Plot Index: {idx}<br>Lat: {garage['lat']}<br>Lng: {garage['lng']}"""
        folium.Marker(location=[float(garage['lat']), float(garage['lng'])], popup=popup_content, icon=folium.Icon(color='red', icon='info-sign')).add_to(marker_cluster)

    for constituency in tqdm(constituencies, desc="Adding constituency boundaries"):
        if isinstance(constituency, geometry.MultiPolygon):
            for sub_poly in constituency.geoms:
                coords = [[y, x] for x, y in sub_poly.exterior.coords]
                folium.Polygon(locations=coords, color='black', weight=1, fill=False).add_to(constituency_layer)
        else:
            coords = [[y, x] for x, y in constituency.exterior.coords]
            folium.Polygon(locations=coords, color='black', weight=1, fill=False).add_to(constituency_layer)

    if road_graph is not None:
        print("Adding traffic data to map...")
        for u, v, data in tqdm(road_graph.edges(data=True), desc="Mapping traffic intensity"):
            if 'traffic_intensity' in data:
                intensity = data['traffic_intensity']
                color_hex = traffic_color(intensity / 2)  # Normalize intensity and map to color

                if 'geometry' in data:
                    coords = [[y, x] for x, y in data['geometry'].coords]
                    folium.PolyLine(locations=coords, color=color_hex, weight=2).add_to(traffic_layer)
                else:
                    start = (road_graph.nodes[u]['y'], road_graph.nodes[u]['x'])
                    end = (road_graph.nodes[v]['y'], road_graph.nodes[v]['x'])
                    folium.PolyLine(locations=[start, end], color=color_hex, weight=2).add_to(traffic_layer)

    marker_cluster.add_to(m)
    voronoi_layer.add_to(m)
    constituency_layer.add_to(m)
    outer_boundary_layer.add_to(m)
    traffic_layer.add_to(m)
    folium.LayerControl().add_to(m)
    plugins.Fullscreen().add_to(m)

    # Add a legend to explain traffic intensity
    legend_html = '''
    <div style="
        position: fixed;
        bottom: 50px; left: 50px; width: 200px; height: 120px;
        background-color: white; z-index:9999; font-size:14px;
        border:2px solid grey; padding: 10px; border-radius: 5px;
    ">
        <b>Traffic Intensity:</b><br>
        <i style="background: green; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> Low<br>
        <i style="background: yellow; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> Medium<br>
        <i style="background: red; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> High
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(filename)


def get_uk_road_graph(filepath="uk_highways.graphml"):
    """Load UK road graph from cache or create and cache it"""
    if os.path.exists(filepath):
        print("Loading cached UK road graph...")
        return ox.load_graphml(filepath)

    print("Downloading and processing UK road graph...")
    cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link"]'
    G = ox.graph_from_place("United Kingdom",
                           network_type="drive",
                           custom_filter=cf,
                           simplify=True,
                           retain_all=False)
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)

    ensure_cache_dir()
    ox.save_graphml(G, filepath)
    return G

def apply_traffic_to_graph(G, gdf_traffic):
    """Apply traffic data to a road network graph"""
    gdf_traffic = gdf_traffic.to_crs(epsg=3857)
    for u, v, k, data in tqdm(G.edges(keys=True, data=True), desc="Applying traffic data"):
        edge_center = ((G.nodes[u]['y'] + G.nodes[v]['y']) / 2, (G.nodes[u]['x'] + G.nodes[v]['x']) / 2)
        edge_center_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([edge_center[1]], [edge_center[0]]), crs='EPSG:4326'
        ).to_crs(epsg=3857)
        nearest_traffic_idx = gdf_traffic.geometry.distance(edge_center_gdf.geometry.iloc[0]).idxmin()
        traffic_intensity = gdf_traffic.loc[nearest_traffic_idx, 'all_motor_vehicles']
        traffic_multiplier = 1 + (traffic_intensity / gdf_traffic['all_motor_vehicles'].max())
        if 'travel_time' in data:
            data['travel_time'] *= traffic_multiplier
        data['traffic_intensity'] = traffic_multiplier
    return G

def load_traffic_data(csv_file_path):
    """Load traffic data from CSV and convert it to a GeoDataFrame"""
    traffic_data = pd.read_csv(csv_file_path)
    traffic_data['coords'] = traffic_data['local_authority_name'].apply(geocode_location)
    traffic_data['latitude'] = traffic_data['coords'].apply(lambda x: x[0] if x is not None else None)
    traffic_data['longitude'] = traffic_data['coords'].apply(lambda x: x[1] if x is not None else None)
    traffic_data.dropna(subset=['latitude', 'longitude'], inplace=True)
    gdf_traffic = gpd.GeoDataFrame(
        traffic_data,
        geometry=gpd.points_from_xy(traffic_data.longitude, traffic_data.latitude),
        crs='EPSG:4326'
    )
    return gdf_traffic

def geocode_location(name):
    """Geocode a location name to its coordinates"""
    try:
        loc = ox.geocoder.geocode(name + ", United Kingdom")
        return loc
    except Exception:
        return None

from shapely.prepared import prep

def main() -> None:
    print("Starting garage mapping...")

    # Load active garages and their data
    active_garages, active_garage_data = get_active_garages()
    active_garages = [geometry.Point(lng, lat) for lng, lat in active_garages]  # Convert to Points
    print(f"Loaded {len(active_garages)} active garages")

    # Load UK boundaries
    outer_boundary, constituencies = get_uk_boundary()
    print("Loaded UK boundaries")

    # Prepare the boundary for efficient spatial queries
    prepared_boundary = prep(outer_boundary)

    # Filter points that are within the boundary
    active_garages = [point for point in active_garages if prepared_boundary.contains(point)]
    print(f"Filtered active garages: {len(active_garages)} remain within the UK boundary")

    # Load and apply traffic data to the road network
    road_graph = get_uk_road_graph()
    traffic_data = load_traffic_data('local_authority_traffic.csv')
    road_graph = apply_traffic_to_graph(road_graph, traffic_data)
    print("Traffic data applied to road network")

    # Integrate traffic data into Voronoi regions
    voronoi_traffic_data = {}
    print("Calculating traffic intensity for Voronoi regions...")
    voronoi_regions, _ = voronoi_regions_from_coords(active_garages, outer_boundary)  # Ensure proper input
    for region_id, poly in voronoi_regions.items():
        intersecting_edges = []
        for u, v, data in road_graph.edges(data=True):
            if 'geometry' in data and poly.intersects(data['geometry']):
                intersecting_edges.append(data['traffic_intensity'])

        if intersecting_edges:
            voronoi_traffic_data[region_id] = sum(intersecting_edges) / len(intersecting_edges)  # Average intensity
        else:
            voronoi_traffic_data[region_id] = 0  # No traffic data for this region

    # Generate and save the interactive map
    create_interactive_map('interactive_garages_map1.html', road_graph=road_graph, voronoi_traffic_data=voronoi_traffic_data)
    print("Interactive map with traffic-enhanced Voronoi regions has been created and saved as 'interactive_garages_map.html'")



if __name__ == "__main__":
    main()


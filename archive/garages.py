import osmnx as ox
import networkx as nx
import json
import os
import folium
from folium import plugins

# Configure OSMnx using settings
ox.settings.log_console = True
ox.settings.use_cache = True

def get_or_create_network():
    """
    Load network from cache if it exists, otherwise create and cache it
    """
    cache_filename = "data/uk_network.graphml"

    if os.path.exists(cache_filename):
        print("Loading cached street network...")
        G = ox.load_graphml(cache_filename)
        return G

    print("Downloading and processing street network...")
    cf = '["highway"~"motorway|trunk|primary|secondary"]'
    G = ox.graph_from_place("United Kingdom",
                           network_type="drive",
                           custom_filter=cf,
                           simplify=True)

    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # Save the processed network
    print("Caching street network...")
    ox.save_graphml(G, cache_filename)

    return G

def create_interactive_map(garages, filename='interactive_garages_map.html'):
    """
    Create an interactive map with garage locations
    """
    # Get the network centre for initial map view
    center_lat = 54.7  # Approximate centre of UK
    center_lng = -3.5

    # Create a folium map
    m = folium.Map(location=[center_lat, center_lng],
                   zoom_start=6,
                   tiles='cartodbpositron')

    # Create a marker cluster group
    marker_cluster = plugins.MarkerCluster().add_to(m)

    # Add markers for active garages
    for garage in garages:
        if ('lat' in garage and 'lng' in garage and
            garage['bookingEnabled']):

            # Create popup content
            popup_content = f"""
                <b>{garage.get('companyName', 'Unknown')}</b><br>
                Lat: {garage['lat']}<br>
                Lng: {garage['lng']}
            """

            # Add marker to cluster
            folium.Marker(
                location=[float(garage['lat']), float(garage['lng'])],
                popup=popup_content,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(marker_cluster)

    # Add fullscreen button
    plugins.Fullscreen().add_to(m)

    # Save the map
    m.save(filename)

def main():
    print("Starting garage mapping...")

    # Read the JSON file
    with open('data/garages.json') as f:
        data = json.load(f)

    # Create interactive map
    print("Creating interactive visualisation...")
    create_interactive_map(data['garageMarkers'])
    print("Map saved as 'interactive_garages_map.html'")

if __name__ == "__main__":
    main()
import numpy as np
import simpy
import random
import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString, Point
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd

VANS_PER_GARAGE = 2 #will set to something very large for actual sim
SIMULATION_TIME = 60*24*5

#G = ox.graph_from_place("Bristol", network_type="drive")
graph = ox.graph_from_place("Bristol", network_type="drive")
edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
edges['geometry'] = edges['geometry'].apply(lambda geom: LineString(geom) if not isinstance(geom, LineString) else geom)
projected_crs = "EPSG:27700"  # choose an appropriate CRS for your area
edges = edges.to_crs(projected_crs)


csvfile= "garage_coords.csv"
df = pd.read_csv(csvfile)
newdf=df[['Latitude','Longitude']]
newdf["location"] = list(zip(df["Latitude"], df["Longitude"]))
newdf["vans"] = [VANS_PER_GARAGE*['available'] for i in range(len(newdf['Latitude']))]
newdf = newdf.drop(columns=["Latitude", "Longitude"])
garages=newdf.to_dict(orient="records")
print(garages)



# Parameters
NUM_GARAGES = 2
#VANS_PER_GARAGE = 3
BREAKDOWN_RATE = 5  # Average time between breakdowns
SERVICE_TIME = 20  # Average time to service a breakdown
#SIMULATION_TIME = 200

# New Parameters

# NUM_GARAGES = imp
# VANS_PER_GARAGE = 1000
# BREAKDOWN_RATE = imp  # Average time between breakdowns
# SERVICE_TIME = 30 + 30*(random.random()-0.5)  # Average time to service a breakdown
# SIMULATION_TIME = 60*24*5


# Function to calculate distance (for simplicity, Euclidean distance)
def distance(lon, lat, start_lon, start_lat):
    target_point = Point(lon, lat)  # Replace with your target coordinates
    target_point = ox.projection.project_geometry(target_point, to_crs=projected_crs)[0]
    closest_edge = edges.distance(target_point).idxmin()
    edge_geometry = edges.loc[closest_edge, 'geometry']
    fraction = np.random.rand()
    partial_distance = fraction * edge_geometry.length
    point_on_edge = edge_geometry.interpolate(partial_distance)
    edges_reset = edges.reset_index()
    u, v = closest_edge[:2]
    filtered_row = edges_reset[(edges_reset['u'] == u) & (edges_reset['v'] == v)]
    if not filtered_row.empty:
        end_node1, end_node2 = filtered_row.iloc[0][['u', 'v']]
    else:
        print(f"No matching edge found for closest_edge: {closest_edge}")

    start_node = ox.nearest_nodes(graph, start_lon, start_lat)
    path1 = nx.shortest_path(graph, start_node, end_node1, weight="length")
    path2 = nx.shortest_path(graph, start_node, end_node2, weight="length")

    distance1 = nx.shortest_path_length(graph, start_node, end_node1, weight="length") + partial_distance
    distance2 = nx.shortest_path_length(graph, start_node, end_node2, weight="length") + (
                edge_geometry.length - partial_distance)

    if distance1 < distance2:
        final_path = path1
        total_distance = distance1
    else:
        final_path = path2
        total_distance = distance2

    return total_distance, final_path


# Breakdown process
def breakdown_generator(env, garages, breakdown_rate):
    while True:
        # Wait for the next breakdown
        yield env.timeout(random.expovariate(1 / breakdown_rate))
        # Generate breakdown location

        #need to work with vandam to make work with voronoi regions
        breakdown_location = (random.uniform(0, 100), random.uniform(0, 100))
        print(f"Breakdown at {breakdown_location} at time {env.now}")
        # Assign a van
        env.process(assign_van(env, garages, breakdown_location))


# Assign a van to a breakdown
def assign_van(env, garages, breakdown_location):
    # need to work with vandam to make work with voronoi regions, should be much quicker than this

    # Find the nearest available van
    nearest_garage = None
    nearest_distance = float('inf')
    for garage in garages:
        if garage['vans'].count('available') > 0:
            dist = distance(garage['location'], breakdown_location)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_garage = garage

    #SERVICE_TIME = norm.rvs(loc=30, scale=8)
    if nearest_garage:
        print(f"Van dispatched from garage at {nearest_garage['location']} to {breakdown_location}")
        # Dispatch a van
        nearest_garage['vans'].remove('available')
        travel_time = nearest_distance / 10  # Assume a speed of 10 units/time
        yield env.timeout(travel_time)
        print(f"Van arrived at breakdown at {env.now}")
        yield env.timeout(SERVICE_TIME)
        print(f"Breakdown serviced at {env.now}")
        # Return the van to the garage
        yield env.timeout(travel_time)
        nearest_garage['vans'].append('available')
        print(f"Van returned to garage at {nearest_garage['location']} at {env.now}")
    else:
        print("No vans available!")


# Simulation setup
def simulate():
    env = simpy.Environment()
    # garages = [
    #     {'location': (10, 10), 'vans': ['available'] * VANS_PER_GARAGE},
    #     {'location': (90, 90), 'vans': ['available'] * VANS_PER_GARAGE},
    # ]

    garages=newdf.to_dict(orient="records")
    env.process(breakdown_generator(env, garages, BREAKDOWN_RATE))
    env.run(until=SIMULATION_TIME)
#
#
# if __name__ == "__main__":
#     simulate()
simulate()

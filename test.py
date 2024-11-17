import osmnx as ox
import networkx as nx
# Configure OSMnx to use the cache and simplify geometries
ox.config(use_cache=True, log_console=True)

# Use a more specific boundary to reduce data size
cf = '["highway"~"motorway|motorway_link|trunk|trunk_link"]'
G = ox.graph_from_place("United Kingdom",
                       network_type="drive",
                       custom_filter=cf,
                       simplify=True,
                       retain_all=False)

G = ox.routing.add_edge_speeds(G)
G = ox.routing.add_edge_travel_times(G)

source_address = "aston expressway"
target_address = "rac uk"

source = ox.geocode(source_address)
target = ox.geocode(target_address)

source_node = ox.nearest_nodes(G, source[1], source[0])
target_node = ox.nearest_nodes(G, target[1], target[0])

route = nx.shortest_path(G, source_node, target_node, weight="travel_time", method="dijkstra")
fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')
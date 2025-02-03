from bayes_opt import BayesianOptimization
from network import Network, VoronoiType
from data import GarageData, TrafficData
from simplified_simulator import SimplifiedSimulator
import logging
import numpy as np
from multiprocessing import Pool, cpu_count
from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from concurrent.futures import ProcessPoolExecutor
import copy

# Generate list of coordinates across constituencies using population data
def generate_breakdown_points(constituencies, constituency_population_data, n_points, network):
    """Generate breakdown points weighted by constituency populations"""
    total_pop = sum(constituency_population_data.values())
    weights = np.array([constituency_population_data[const]/total_pop for const in constituencies])
    weights = weights / weights.sum()

    simulator = SimplifiedSimulator(network)
    simulator.constituencies = constituencies
    simulator.population_weights = weights.tolist()

    breakdown_points = []
    for _ in range(n_points):
        _, coords = simulator._random_constituency_point()
        breakdown_points.append(coords)

    return breakdown_points

## New global variable and worker initializer for ProcessPool tasks
global_network = None

def worker_init(network_arg):
    global global_network
    global_network = network_arg

def analyse_single_route_worker(coord, i, voronoiType: VoronoiType):
    """Worker function using a global network variable."""
    return analyse_single_route(global_network, coord, i, voronoiType)

def analyse_single_route(network: Network, coord, i, voronoiType: VoronoiType):
    """Analyze a single route with caching of road network"""
    # Use cached road network to avoid reloading for each route
    road_network = network.get_road_network()

    route_analysis = network.analyse_route(coord, voronoiType, 8, cached_network=road_network)
    if route_analysis is None:
        logging.warning(f"Could not analyse route {i} for coordinate {coord}")
        return None

    raw_travel_time = route_analysis['travel_time']
    travel_time = (raw_travel_time * 2) + 30

    logging.info(f"Route {i} analyzed: {travel_time:.1f} mins")
    return travel_time

def get_average_response_time(network: Network, breakdown_coords, voronoiType: VoronoiType):
    """Calculate average response time using parallel processing with improvements"""
    travel_times = []

    # Pre-cache the road network (invariant across evaluations)
    road_network = network.get_road_network()
    network._road_network = road_network

    # Use a fixed number of processes; note that too many processes may add overhead.
    n_workers = min(os.cpu_count(), len(breakdown_coords))
    logging.info(f"Getting average response time with {n_workers} workers using ProcessPoolExecutor")

    # Use ProcessPoolExecutor with worker initializer to share the network object.
    with ProcessPoolExecutor(max_workers=n_workers, initializer=worker_init, initargs=(network,)) as executor:
        # Submit all coordinates to process pool using the worker function that uses the global network.
        futures = [
            executor.submit(analyse_single_route_worker, coord, i, voronoiType)
            for i, coord in enumerate(breakdown_coords)
        ]

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    travel_times.append(result)
            except Exception as e:
                logging.error(f"Error analyzing route: {e}")

    if not travel_times:
        return float('inf')

    avg_time = sum(travel_times) / len(travel_times)
    logging.info(f"Finished getting average response time: {avg_time:.1f} mins ({len(travel_times)} valid routes)")
    return avg_time

def get_network_boundary_limits(network: Network) -> tuple[dict, tuple[float, float, float, float]]:
    """Get the coordinate limits and center point of the network boundary"""
    bounds = network.boundary.bounds

    # Calculate center point and dimensions
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    # Calculate max radius (in degrees) from center to ensure we stay within network
    max_radius = min(
        abs(bounds[3] - bounds[1]) / 2,  # latitude radius
        abs(bounds[2] - bounds[0]) / 2   # longitude radius
    ) * 0.8  # 80% of max radius to stay well within bounds

    limits = {
        'r': (0, max_radius),     # radius from center
        'theta': (0, 2 * np.pi)   # angle in radians
    }

    return limits, (center_lat, center_lon, max_radius, max_radius)

def polar_to_cartesian(r: float, theta: float, center_lat: float, center_lon: float) -> tuple[float, float]:
    """Convert polar coordinates to latitude/longitude"""
    lat = center_lat + r * np.cos(theta)
    lon = center_lon + r * np.sin(theta)
    return lat, lon

def check_coord_inside_network(network: Network, x: float, y: float) -> bool:
    point = Point(y, x)
    return network.boundary.contains(point)

def analyse_single_breakdown(network: Network, coord: tuple, hour: int):
    """Analyze a single breakdown point to find nearest garage"""
    nearest = network.find_nearest_garage(coord, hour)
    if nearest and 'garage' in nearest and 'Postcode' in nearest['garage']:
        return nearest['garage']['Postcode']
    return None

def get_garage_breakdown_distribution(network: Network, breakdown_coords: list, trafficHour) -> tuple[dict, float]:
    """Calculate the distribution of breakdowns across garages using parallel processing"""
    # Initialize counters for all garages
    breakdown_counts = {garage['Postcode']: 0 for _, garage in network.filtered_garages.iterrows()}

    # Pre-cache the road network
    road_network = network.get_road_network()
    network._road_network = road_network

    # Use a fixed number of processes
    n_workers = min(os.cpu_count(), len(breakdown_coords))
    logging.info(f"Calculating breakdown distribution with {n_workers} workers")

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all coordinates to process pool
        futures = [
            executor.submit(analyse_single_breakdown, network, coord, trafficHour)
            for coord in breakdown_coords
        ]

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                garage_id = future.result()
                if garage_id is not None:
                    breakdown_counts[garage_id] += 1
            except Exception as e:
                logging.error(f"Error analyzing breakdown: {e}")

    # Calculate distribution score
    total_breakdowns = len(breakdown_coords)
    if total_breakdowns == 0:
        return breakdown_counts, 0.0

    # Expected even distribution
    num_garages = len(network.filtered_garages)
    expected_per_garage = total_breakdowns / num_garages

    # Calculate coefficient of variation (CV)
    counts = np.array(list(breakdown_counts.values()))
    mean = np.mean(counts)
    std = np.std(counts)
    cv = std / mean if mean > 0 else float('inf')

    # Convert to a score between 0 and 1
    # When CV = 0, distribution is perfect (score = 1)
    # When CV is large, score approaches 0
    distribution_score = 1 / (1 + cv)

    logging.info(f"Calculated distribution score: {distribution_score:.3f}")
    return breakdown_counts, distribution_score

def cost_function(network: Network, original_garages: dict, garages_to_move: list,
                 breakdown_points: list, center_coords: tuple, trafficHour, **garage_coords) -> float:
    """
    Calculate cost for potential garage locations using both response time and distribution
    """
    temp_garage_data = GarageData()
    temp_garage_data.df = original_garages.df.copy()
    center_lat, center_lon, _, _ = center_coords

    logging.info("Updating garage coordinates")
    for postcode in garages_to_move:
        r = garage_coords[f'{postcode}_r']
        theta = garage_coords[f'{postcode}_theta']

        # Convert polar to cartesian coordinates
        lat, lon = polar_to_cartesian(r, theta, center_lat, center_lon)

        # Check if coordinates are valid
        point = Point(lon, lat)
        if not network.boundary.contains(point):
            distance = point.distance(network.boundary)
            return -1000 - (distance * 1000)

        # Update coordinates in both DataFrame and GeoDataFrame
        temp_garage_data.update_garage_location(postcode, lat, lon)

    # Create temporary network with updated garage locations and traffic Voronoi
    temp_network = Network(
        temp_garage_data,
        network.constituencies,
        traffic_hour=trafficHour,
        use_voronoi_cache=False
    )

    # temp_network.show_network(show_garages=True, show_garage_labels=True, voronoi_type=VoronoiType.EUCLIDEAN)
    logging.info("Finished creating network with new garage coordinates")

    # Calculate average response time
    avg_response_time = get_average_response_time(temp_network, breakdown_points, VoronoiType.TRAFFIC)

    # Calculate distribution score
    breakdown_counts, distribution_score = get_garage_breakdown_distribution(temp_network, breakdown_points, trafficHour)

    logging.info(f"Average response time: {avg_response_time:.1f} mins")
    logging.info(f"Distribution score: {distribution_score:.3f}")
    logging.info(f"Breakdown distribution: {breakdown_counts}")

    # Combine metrics (negative because we want to maximize)
    response_time_weight = 0.7
    distribution_weight = 0.3

    # Normalize response time (assuming typical range 0-60 minutes)
    normalized_time = max(0, min(1, avg_response_time / 60))

    combined_score = (
        -normalized_time * response_time_weight +
        distribution_score * distribution_weight
    )

    return combined_score

def main():
    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logging.getLogger('matplotlib').setLevel(logging.INFO)

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

    voronoiType = VoronoiType.TRAFFIC
    traffic_hour = 8

    traffic_data = TrafficData()
    garage_data = GarageData()
    network = Network(
        garage_data,
        constituencies,
        traffic_hour=traffic_hour,
        use_voronoi_cache=False
    )

    constituency_population_data = network.constituency_populations
    number_of_breakdowns = 7

    garages_to_move = ['GL2 5DQ', 'BS5 6NJ']

    # Generate breakdown points once
    logging.info("Generating breakdown points")
    breakdown_points = generate_breakdown_points(
        constituencies,
        constituency_population_data,
        number_of_breakdowns,
        network
    )
    logging.info("Finished generating breakdown points")

    logging.info("Getting network boundary")
    bounds, center_coords = get_network_boundary_limits(network)
    pbounds = {}
    for postcode in garages_to_move:
        pbounds[f'{postcode}_r'] = bounds['r']        # radius bounds
        pbounds[f'{postcode}_theta'] = bounds['theta'] # angle bounds
    logging.info("Finished getting network boundary")

    initial_response_time = get_average_response_time(network, breakdown_points, VoronoiType.TRAFFIC)
    print(f"\nInitial average response time (mins): {initial_response_time}\n")

    initial_distribution_score = get_garage_breakdown_distribution(network, breakdown_points, traffic_hour)
    print(f"Initial distribution score: {initial_distribution_score[1]}\n")

    optimizer = BayesianOptimization(
        f=lambda **kwargs: cost_function(network, garage_data, garages_to_move,
                                       breakdown_points, center_coords, traffic_hour, **kwargs),
        pbounds=pbounds,
        random_state=1
    )

    # Run optimization
    optimizer.maximize(
        init_points=2,
        n_iter=5
    )

    # Convert best result back to lat/lon for display
    best_params = optimizer.max['params']
    best_locations = {}
    for postcode in garages_to_move:
        r = best_params[f'{postcode}_r']
        theta = best_params[f'{postcode}_theta']
        lat, lon = polar_to_cartesian(r, theta, center_coords[0], center_coords[1])
        best_locations[postcode] = (lat, lon)

    print("\nBest locations:", best_locations)

    # Create garage data with updated coords
    updated_garage_data = copy.deepcopy(garage_data)
    for postcode, (lat, lon) in best_locations.items():
        # Find index of garage with matching postcode
        garage_idx = updated_garage_data.gdf[updated_garage_data.gdf['Postcode'] == postcode].index[0]
        # Update coordinates
        updated_garage_data.gdf.at[garage_idx, 'geometry'] = Point(lon, lat)

    best_network = Network(updated_garage_data, constituencies, traffic_hour=traffic_hour, use_voronoi_cache=False)

    final_response_time = get_average_response_time(best_network, breakdown_points, VoronoiType.TRAFFIC)
    print(f"\nFinal average response time (mins): {final_response_time}\n")

    final_distribution_score = get_garage_breakdown_distribution(best_network, breakdown_points, traffic_hour)
    print(f"Final distribution score: {final_distribution_score[1]}\n")

    best_network.show_network(show_garages=True, show_garage_labels=True, voronoi_type=VoronoiType.TRAFFIC, traffic_hour=traffic_hour)


if __name__ == "__main__":
    main()
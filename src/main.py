from data import TrafficData, GarageData
from network import Network, VoronoiType
from simulator import BreakdownSimulator
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# ---

traffic_data = TrafficData()

# Traffic data across the 2023 year (interactive plot)
# traffic_data.plot_daily_traffic_counts()

# Average traffic data across the 2023 year (static plot) (used for traffic factors)
# traffic_data.plot_average_traffic_counts()

garage_data = GarageData()

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

if __name__ == "__main__":
    # Initialize network
    bristol_network = Network(garage_data, constituencies)

    # Run visualization
    bristol_network.show_network(
        show_garages=True,
        show_roads=True,
        show_constituencies=True,
        show_traffic=False,
        voronoi_type=VoronoiType.NETWORK,
        traffic_hour=8,
        # coord=(51.574083, -2.616579)
    )

    # Run simulation
    # simulator = BreakdownSimulator(
    #     network=bristol_network,
    #     vans_per_garage=2
    # )
    # simulator.run(
    #     simulation_time=60*8*1,  # 5 days
    #     breakdown_rate=5  # Average time between breakdowns in minutes
    # )

# bristol_network.show_network(
#     show_garages=True,
#     show_roads=True,
#     show_traffic=False,
#     voronoi_type=VoronoiType.NETWORK,
#     traffic_hour=8,
#     coord=(51.574083, -2.616579)
# )

# bristol_network.show_network(
#     show_garages=True,
#     show_roads=True,
#     show_traffic=False,
#     voronoi_type=VoronoiType.TRAFFIC,
#     traffic_hour=8,
#     coord=(51.574083, -2.616579)
# )

# Test coordinate lookup with route analysis
# coord = (51.574083, -2.616579)
# for voronoi_type in [VoronoiType.EUCLIDEAN, VoronoiType.TRAFFIC, VoronoiType.NETWORK]:
#     print(f"\nAnalysing {voronoi_type.value} route:")
#     analysis = bristol_network.analyse_route(
#         coord,
#         voronoi_type,
#         hour=8 if voronoi_type == VoronoiType.TRAFFIC else None
#     )

#     if analysis:
#         garage = analysis['garage']
#         print(f"From {garage.get('Postcode', 'No Postcode')}:")
#         print(f"Travel time: {analysis['travel_time']} mins")
#         distance_km = analysis['distance']
#         distance_mi = distance_km * 0.621371
#         print(f"Distance: {distance_km:.1f} km ({distance_mi:.1f} mi)")
#         avg_speed_kmh = distance_km/(analysis['travel_time']/60)
#         avg_speed_mph = avg_speed_kmh * 0.621371
#         print(f"Average speed: {avg_speed_kmh:.1f} km/h ({avg_speed_mph:.1f} mph)")
#     else:
#         print("No route found")

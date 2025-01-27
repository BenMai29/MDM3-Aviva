from data import TrafficData, GarageData
from network import Network, VoronoiType
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# ---

traffic_data = TrafficData()

# Traffic data across the 2023 year (interactive plot)
traffic_data.plot_daily_traffic_counts()

# Average traffic data across the 2023 year (static plot) (used for traffic factors)
traffic_data.plot_average_traffic_counts()

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

bristol_network = Network(garage_data, constituencies)

# for hour in [8, 10, 12, 14, 16, 18]:
#     bristol_network.show_network(
#         show_garages=True,
#         show_roads=True,
#         show_traffic=True,
#         voronoi_type=VoronoiType.TRAFFIC,
#         traffic_hour=hour
#     )

bristol_network.show_network(
    show_garages=True,
    show_roads=True,
    show_traffic=True,
    voronoi_type=VoronoiType.TRAFFIC,
    traffic_hour=8,
    coord=(51.574083, -2.616579)
)

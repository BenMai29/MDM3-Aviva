from data import TrafficData, GarageData
from network import Network, VoronoiType
from simplified_simulator import SimplifiedSimulator
from simulator import BreakdownSimulator
import logging
from visualisation import AnimatedResults

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logging.getLogger('matplotlib').setLevel(logging.WARNING)

traffic_data = TrafficData()
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
    bristol_network = Network(garage_data, constituencies, use_voronoi_cache=False)
    bristol_network.show_network(show_garages=True, show_garage_labels=True, voronoi_type=VoronoiType.TRAFFIC, traffic_hour=8, show_roads=True)


    # simulator = BreakdownSimulator(
    #     bristol_network,
    #     voronoi_type=VoronoiType.TRAFFIC
    # )
    # results = simulator.run(
    #     simulation_days=1,
    #     breakdown_rate=8.17
    # )

    # print(f"Simulated {len(results)} breakdowns")

    # # scene = AnimatedResults(
    # #     results_path="results/simulation_results_20250129_230927_run1.json",
    # #     start_time=420,  # 7:00
    # #     end_time=540,  # 9:00
    # # )

    # scene.render()

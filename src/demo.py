from data import TrafficData, GarageData
from network import Network, VoronoiType
from simulator import BreakdownSimulator

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
    bristol_network = Network(garage_data, constituencies)

    # Run simulation with 5 parallel runs
    simulator = BreakdownSimulator(
        network=bristol_network,
        vans_per_garage=200,
        voronoi_type=VoronoiType.TRAFFIC
    )
    simulator.run(
        simulation_days=1,
        breakdown_rate=8.165342563671928,
        num_runs=5
    )
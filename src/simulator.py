import simpy
import random
import numpy as np
from network import Network
from typing import List
from shapely.geometry import Point
from network import VoronoiType
from datetime import datetime
from pathlib import Path
import multiprocessing
from functools import partial
import json
import shapely

def format_time(minutes):
    """Convert minutes to days, hours, minutes format"""
    days = minutes // (24 * 60)
    remaining = minutes % (24 * 60)
    hours = remaining // 60
    mins = remaining % 60
    return f"Day {days}, {hours:02d}:{mins:02d}"

class BreakdownSimulator:
    SERVICE_TIME = 30  # Average service time in minutes
    SIMULATION_START_HOUR = 7  # 07:00
    SIMULATION_END_HOUR = 18   # 18:00

    def __init__(self, network: Network, vans_per_garage: int = 2, voronoi_type: VoronoiType = VoronoiType.NETWORK):
        self.network = network
        self.vans_per_garage = vans_per_garage
        self.voronoi_type = voronoi_type  # Store voronoi type
        self.garages = self._init_garages()
        self.breakdown_probabilities = self._calculate_breakdown_probabilities()
        # Add statistics tracking
        self.garage_stats = {
            garage['id']: {
                'total_breakdowns': 0,
                'total_time': 0,  # Sum of all service times (including travel and repair)
                'max_vans_out': 0,  # Maximum number of vans out simultaneously
                'current_vans_out': 0  # Current number of vans out
            }
            for garage in self.garages
        }
        self.last_time_display = 0  # Track last displayed time
        # Cache the road network
        self._road_network = self.network.get_road_network()
        self.breakdown_events = []  # Add this line

    def _init_garages(self):
        """Initialize garage states based on network data"""
        return [{
            'id': row['Postcode'],  # Use postcode instead of index
            'location': (row.geometry.y, row.geometry.x),
            'vans': ['available'] * self.vans_per_garage
        } for _, row in self.network.filtered_garages.iterrows()]

    def _calculate_breakdown_probabilities(self):
        """Calculate constituency probabilities based on population"""
        total_pop = sum(self.network.constituency_populations.values())
        return {
            const: pop / total_pop
            for const, pop in self.network.constituency_populations.items()
        }

    def _generate_breakdown(self, env):
        """Generate a random breakdown location and time"""
        # Get road network boundary
        road_network = self.network.get_road_network()
        nodes = list(road_network.nodes(data=True))  # Convert to list of (node_id, attributes)

        # Get random node from road network
        node_id, node_data = random.choice(nodes)
        lon, lat = node_data['x'], node_data['y']  # x=longitude, y=latitude

        # Get current hour for traffic analysis
        current_time = env.now
        hour = (current_time // 60) % 24

        # Find nearest garage - pass coordinates as (latitude, longitude)
        analysis = self.network.analyse_route((lat, lon), voronoi_type=self.voronoi_type, hour=hour)
        if not analysis:
            return None

        return {
            'time': int(current_time),
            'location': {
                'type': 'Point',
                'coordinates': [lon, lat]  # GeoJSON uses [longitude, latitude]
            },
            'garage_id': analysis['garage']['Postcode']
        }

    def breakdown_generator(self, env, breakdown_rate: float):
        """Generate breakdown events only during working hours"""
        while True:
            current_time = int(env.now)
            current_hour = (current_time // 60) % 24

            # Only generate breakdowns during working hours
            if self.SIMULATION_START_HOUR <= current_hour < self.SIMULATION_END_HOUR:
                # if hasattr(self, 'show_progress') and self.show_progress:
                    # if current_time >= self.last_time_display + 30:
                        # print(f"\rSimulation time: {format_time(current_time)}\n", end="", flush=True)
                        # self.last_time_display = current_time

                yield env.timeout(random.expovariate(1 / breakdown_rate))
                breakdown_point = self._generate_breakdown(env)
                if breakdown_point:  # Only process valid breakdowns
                    env.process(self.assign_van(env, breakdown_point))
            else:
                # Skip to next working day if outside working hours
                minutes_until_next_day = ((24 - current_hour + self.SIMULATION_START_HOUR) % 24) * 60
                yield env.timeout(minutes_until_next_day)

    def assign_van(self, env, breakdown_point):
        """Optimized van assignment"""
        current_time = int(env.now)
        coord = (breakdown_point['location']['coordinates'][1], breakdown_point['location']['coordinates'][0])

        # Create event dictionary at start of assignment
        event = {
            'time': current_time,
            'simulation_time': format_time(current_time),
            'location': breakdown_point['location'],
            'garage_id': breakdown_point['garage_id'],
            'travel_time': None,
            'route_geometry': None,
            'total_job_time': None,
            'status': 'dispatched'
        }

        garage_data = self.network.find_garage_for_coordinate(
            coord,
            self.voronoi_type,
            hour=current_time // 60 if self.voronoi_type == VoronoiType.TRAFFIC else None
        )

        if not garage_data:
            print("No responsible garage found!")
            return

        garage = next(g for g in self.garages if g['id'] == garage_data['Postcode'])
        garage_stats = self.garage_stats[garage['id']]

        if not garage['vans']:
            print(f"No vans available at garage {garage['id']}!")
            return

        # Update statistics
        garage_stats['total_breakdowns'] += 1
        garage_stats['current_vans_out'] += 1
        garage_stats['max_vans_out'] = max(
            garage_stats['max_vans_out'],
            garage_stats['current_vans_out']
        )

        # Dispatch van
        garage['vans'].pop()
        print(f"Van dispatched from garage {garage['id']} at {format_time(current_time)}")

        # Use cached route analysis with current hour
        analysis = self.network.analyse_route(
            coord,
            self.voronoi_type,
            hour=current_time // 60 if self.voronoi_type == VoronoiType.TRAFFIC else None,
            cached_network=self._road_network
        )

        if not analysis:
            event['status'] = 'failed'
            self.breakdown_events.append(event)
            return

        # Remove conversion - use travel_time directly as minutes
        travel_time = analysis['travel_time']

        # Calculate total job time (now in minutes)
        total_job_time = 2 * travel_time + self.SERVICE_TIME
        garage_stats['total_time'] += total_job_time

        # Simulate travel to breakdown (1:1 minute scale)
        yield env.timeout(travel_time)
        arrival_time = int(env.now)
        print(f"Van from {garage['id']} arrived at breakdown at {format_time(arrival_time)}")

        # Service time remains in minutes
        yield env.timeout(self.SERVICE_TIME)

        # Return trip
        yield env.timeout(travel_time)
        return_time = int(env.now)
        print(f"Van returned to garage {garage['id']} at {format_time(return_time)}")

        # Van returns
        garage['vans'].append('available')
        garage_stats['current_vans_out'] -= 1

        # Update event with actual timestamps
        event.update({
            'travel_time': travel_time,
            'route_geometry': shapely.geometry.mapping(analysis['route_geometry']),
            'total_job_time': return_time - current_time,
            'status': 'completed'
        })

        self.breakdown_events.append(event)  # Add this line

    def print_statistics(self):
        """Print final statistics for each garage"""
        print("\n=== Simulation Statistics ===")
        for garage_id, stats in self.garage_stats.items():
            if stats['total_breakdowns'] > 0:
                avg_time = stats['total_time'] / (stats['total_breakdowns'] * 60)  # Convert to minutes
                print(f"\nGarage {garage_id}:")
                print(f"  Total breakdowns: {stats['total_breakdowns']}")
                print(f"  Average total job time: {avg_time:.1f} minutes")
                print(f"  Maximum vans out simultaneously: {stats['max_vans_out']}")
            else:
                print(f"\nGarage {garage_id}: No breakdowns")

    def save_statistics(self, filename: str = "simulation_results.txt", run_num: int = None):
        """Save statistics to a text file"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Use class-level timestamp if it exists, otherwise create new one
        if not hasattr(BreakdownSimulator, '_timestamp'):
            BreakdownSimulator._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if run_num is not None:
            filepath = results_dir / f"simulation_results_{BreakdownSimulator._timestamp}_run{run_num}.txt"
        else:
            filepath = results_dir / f"simulation_results_{BreakdownSimulator._timestamp}.txt"

        with open(filepath, 'w') as f:
            f.write("=== Simulation Results ===\n")
            f.write(f"Run number: {run_num if run_num is not None else 1}\n")
            f.write(f"Voronoi type: {self.voronoi_type.value}\n\n")

            for garage_id, stats in self.garage_stats.items():
                f.write(f"Garage {garage_id}:\n")
                if stats['total_breakdowns'] > 0:
                    avg_time = stats['total_time'] / (stats['total_breakdowns'] * 60)
                    f.write(f"  Total breakdowns: {stats['total_breakdowns']}\n")
                    f.write(f"  Average total job time: {avg_time:.1f} minutes\n")
                    f.write(f"  Maximum vans out simultaneously: {stats['max_vans_out']}\n")
                else:
                    f.write("  No breakdowns handled\n")
                f.write("\n")

        print(f"\nResults saved to {filepath}")

        # Add this section to save events as JSON
        events_filepath = filepath.with_suffix('.json')
        with open(events_filepath, 'w') as f:
            json.dump(self.breakdown_events, f, indent=2, default=str)

        print(f"\nEvents saved to {events_filepath}")

    def _single_run(self, run_num: int, simulation_days: int, breakdown_rate: float, show_logs: bool = False):
        """Run a single simulation"""
        if show_logs:
            print(f"\n=== Run {run_num}/{self.num_runs} ===")
            print(f"Daily hours: {self.SIMULATION_START_HOUR:02d}:00 - {self.SIMULATION_END_HOUR:02d}:00")

        # Initialize fresh stats for this run
        garage_stats = {
            garage['id']: {
                'total_breakdowns': 0,
                'total_time': 0,
                'max_vans_out': 0,
                'current_vans_out': 0
            }
            for garage in self.garages
        }

        # Fresh vans for this run
        garages = [{**g, 'vans': ['available'] * self.vans_per_garage} for g in self.garages]

        # Calculate simulation time
        minutes_per_day = (self.SIMULATION_END_HOUR - self.SIMULATION_START_HOUR) * 60
        total_simulation_time = minutes_per_day * simulation_days

        # Create new environment for this run
        env = simpy.Environment(initial_time=self.SIMULATION_START_HOUR * 60)

        # Create a new simulator instance for this run to avoid shared state
        sim = BreakdownSimulator(self.network, self.vans_per_garage, self.voronoi_type)
        sim.garage_stats = garage_stats
        sim.garages = garages
        sim.last_time_display = 0

        # Only show progress for first run
        sim.show_progress = show_logs

        # Run simulation
        env.process(sim.breakdown_generator(env, breakdown_rate))
        env.run(until=total_simulation_time + (self.SIMULATION_START_HOUR * 60))

        # Save results
        sim.save_statistics(run_num=run_num)

        if show_logs:
            sim.print_statistics()

        return sim.garage_stats

    @classmethod
    def _worker(cls, run_num: int, show_logs: bool, simulator, simulation_days: int, breakdown_rate: float):
        """Worker function for parallel simulation runs"""
        return simulator._single_run(
            run_num=run_num,
            simulation_days=simulation_days,
            breakdown_rate=breakdown_rate,
            show_logs=show_logs
        )

    def run(self, simulation_days: int = 5, breakdown_rate: float = 5, num_runs: int = 1):
        """Run multiple simulations in parallel

        Args:
            simulation_days: Number of days to simulate
            breakdown_rate: Average time between breakdowns in minutes
            num_runs: Number of simulation runs to perform
        """
        self.num_runs = num_runs  # Store for use in _single_run
        print(f"Starting {num_runs} parallel simulation runs")
        print(f"Parameters: {simulation_days} days, {breakdown_rate} min breakdown rate")

        # Prepare run configurations with all necessary parameters
        run_configs = [
            (i+1, i == 0, self, simulation_days, breakdown_rate)
            for i in range(num_runs)
        ]

        # Run simulations in parallel
        with multiprocessing.Pool() as pool:
            pool.starmap(self._worker, run_configs)
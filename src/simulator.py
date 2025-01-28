import simpy
import random
import numpy as np
from network import Network
from typing import List
from shapely.geometry import Point
from network import VoronoiType

def format_time(minutes):
    """Convert minutes to days, hours, minutes format"""
    days = minutes // (24 * 60)
    remaining = minutes % (24 * 60)
    hours = remaining // 60
    mins = remaining % 60
    return f"Day {days}, {hours:02d}:{mins:02d}"

class BreakdownSimulator:
    SERVICE_TIME = 30  # Average service time in minutes

    def __init__(self, network: Network, vans_per_garage: int = 2):
        self.network = network
        self.vans_per_garage = vans_per_garage
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

    def _generate_breakdown_location(self):
        """Generate breakdown location weighted by constituency population"""
        # Select constituency
        constituency = random.choices(
            list(self.breakdown_probabilities.keys()),
            weights=list(self.breakdown_probabilities.values()),
            k=1
        )[0]

        # Generate point within constituency boundary
        boundary = self.network.get_constituency_boundary(constituency)
        minx, miny, maxx, maxy = boundary.bounds

        while True:
            point = Point(
                random.uniform(minx, maxx),
                random.uniform(miny, maxy)
            )
            if point.within(boundary):
                return point

    def breakdown_generator(self, env, breakdown_rate: float):
        """Generate breakdown events"""
        while True:
            current_time = int(env.now)
            if current_time >= self.last_time_display + 30:
                print(f"\rSimulation time: {format_time(current_time)}", end="", flush=True)
                self.last_time_display = current_time

            yield env.timeout(random.expovariate(1 / breakdown_rate))
            breakdown_point = self._generate_breakdown_location()
            env.process(self.assign_van(env, breakdown_point))

    def assign_van(self, env, breakdown_point):
        """Assign van using network Voronoi regions"""
        garage_data = self.network.find_garage_for_coordinate(
            (breakdown_point.y, breakdown_point.x),
            VoronoiType.NETWORK
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
        print(f"Van dispatched from garage {garage['id']} at {env.now}")

        analysis = self.network.analyse_route(
            (breakdown_point.y, breakdown_point.x),
            VoronoiType.NETWORK
        )

        if not analysis:
            print("No route found!")
            garage['vans'].append('available')
            garage_stats['current_vans_out'] -= 1
            return

        # Calculate total job time
        total_job_time = (2 * analysis['travel_time'] + self.SERVICE_TIME) * 60
        garage_stats['total_time'] += total_job_time

        # Simulate travel and service
        yield env.timeout(analysis['travel_time'] * 60)
        print(f"Van arrived at breakdown at {env.now}")
        yield env.timeout(self.SERVICE_TIME)
        print(f"Service completed at {env.now}")
        yield env.timeout(analysis['travel_time'] * 60)

        # Van returns
        garage['vans'].append('available')
        garage_stats['current_vans_out'] -= 1
        print(f"Van returned to garage {garage['id']} at {env.now}")

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

    def run(self, simulation_time: int = 60*24*5, breakdown_rate: float = 5):
        """Run the simulation"""
        print(f"Starting simulation for {format_time(simulation_time)}")
        env = simpy.Environment()
        env.process(self.breakdown_generator(env, breakdown_rate))
        env.run(until=simulation_time)
        print("\n")  # New line after time display
        self.print_statistics()
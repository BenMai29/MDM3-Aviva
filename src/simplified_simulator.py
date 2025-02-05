import logging
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List
from network import Network, VoronoiType
from shapely.geometry import Point

class SimplifiedSimulator:
    WORKING_HOURS = (7, 18)  # 7am to 6pm
    MINUTES_PER_DAY = (WORKING_HOURS[1] - WORKING_HOURS[0]) * 60

    def __init__(self, network: Network, voronoi_type: VoronoiType = VoronoiType.NETWORK):
        logging.info("Initialising SimplifiedSimulator")
        self.network = network
        self.voronoi_type = voronoi_type
        self.breakdown_events = []
        self.constituencies = list(network._constituency_boundaries.keys())
        self.population_weights = self._get_population_weights()
        self._road_network = network.get_road_network()
        self.cached_breakdown_points = None
        logging.debug(f"Initialised with {len(self.constituencies)} constituencies")

    def _get_population_weights(self):
        """Get normalised population weights for constituencies"""
        logging.debug("Calculating population weights")
        populations = self.network.load_constituency_populations()
        total = sum(populations.values())
        weights = [populations[const]/total for const in self.constituencies]
        logging.debug(f"Population weights calculated: {dict(zip(self.constituencies, weights))}")
        return weights

    def run(self, simulation_days: int = 1, breakdown_rate: float = 8.17):
        """Run simulation and return breakdown events.
           If cached breakdown points exist, reuse them.
        """
        logging.info(f"Starting simulation for {simulation_days} days with breakdown rate {breakdown_rate}")

        # Generate or reuse cached breakdown points
        if self.cached_breakdown_points is None:
            self.cached_breakdown_points = []
            breakdown_times = self._generate_breakdown_times(simulation_days, breakdown_rate)
            for t in breakdown_times:
                constituency, coords = self._random_constituency_point()
                self.cached_breakdown_points.append((t, constituency, coords))

        # Use the cached breakdown points for simulation events
        i = 0
        self.breakdown_events = []  # reset for this run
        for (timestamp, constituency, coords) in self.cached_breakdown_points:
            logging.info(f"{i}/{len(self.cached_breakdown_points)}: Generated breakdown at {coords} in {constituency} at {self._format_time(timestamp)}")
            analysis = self._analyze_breakdown(coords, timestamp)
            self.breakdown_events.append({
                'timestamp': timestamp,
                'constituency': constituency,
                'coordinates': coords,
                'formatted_time': self._format_time(timestamp),
                'route_analysis': analysis
            })
            logging.debug(f"Added breakdown event at {self._format_time(timestamp)}")
            i += 1

        self._save_results()
        logging.info(f"Simulation complete with {len(self.breakdown_events)} events")
        return self.breakdown_events

    def _generate_breakdown_times(self, days: int, rate: float):
        """Generate sorted list of breakdown timestamps"""
        logging.debug(f"Generating breakdown times for {days} days")
        times = sorted(
            self._generate_breakdown_time(day)
            for day in range(days)
            for _ in range(int(self.MINUTES_PER_DAY / rate))
        )
        return times

    def _generate_breakdown_time(self, day: int) -> int:
        """Generate random minute within working hours for a day"""
        time = day * 24 * 60 + self.WORKING_HOURS[0] * 60 + random.randint(0, self.MINUTES_PER_DAY)
        logging.debug(f"Generated breakdown time: {self._format_time(time)}")
        return time

    def _analyze_breakdown(self, coords: tuple, timestamp: int) -> dict:
        """Analyse route for a breakdown event"""
        logging.debug(f"Analysing breakdown at coordinates {coords}")
        hour = (timestamp // 60) % 24
        try:
            analysis = self.network.analyse_route(
                coords,
                self.voronoi_type,
                hour=hour if self.voronoi_type == VoronoiType.TRAFFIC else None,
                cached_network=self._road_network
            )
            if analysis:
                logging.debug(f"Route analysis successful: garage={analysis['garage']['Postcode']}, travel_time={analysis['travel_time']}")
                return {
                    'garage': analysis['garage']['Postcode'],
                    'travel_time': analysis['travel_time'],
                    'distance_km': analysis['distance'],
                    'route_geometry': analysis['route_geometry'].wkt
                }
        except Exception as e:
            logging.error(f"Route analysis failed: {str(e)}")
        return None

    def _random_constituency_point(self) -> tuple:
        """Select constituency weighted by population and generate random point"""
        # Weighted selection based on population
        constituency = np.random.choice(
            self.constituencies,
            p=self.population_weights
        )
        logging.debug(f"Selected constituency: {constituency}")

        # Generate random point within constituency boundary
        boundary = self.network.get_constituency_boundary(constituency)
        minx, miny, maxx, maxy = boundary.bounds

        # Generate random point within bounds until one falls inside the polygon
        attempts = 0
        while True:
            attempts += 1
            point = Point(
                random.uniform(minx, maxx),
                random.uniform(miny, maxy)
            )
            if boundary.contains(point):
                logging.debug(f"Generated valid point after {attempts} attempts")
                return constituency, (point.y, point.x)  # (lat, lng)

    def _format_time(self, minutes: int) -> str:
        """Format minutes into simulation time string"""
        days = minutes // (24 * 60)
        remaining = minutes % (24 * 60)
        return f"Day {days}, {remaining//60:02d}:{remaining%60:02d}"

    def _save_results(self):
        """Save results to JSON file"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output = {
            'parameters': {
                'constituencies': self.constituencies,
                'total_breakdowns': len(self.breakdown_events)
            },
            'events': self.breakdown_events
        }

        file_path = results_dir / f"breakdowns_{timestamp}.json"
        with open(file_path, 'w') as f:
            json.dump(output, f, indent=2)

        logging.info(f"Saved results to {file_path}")

    def evaluate_average_response_time(self, simulation_days: int = 1, breakdown_rate: float = 8.17) -> float:
        """
        Run the simulation (reusing breakdown points) and return the average response time (in minutes).
        If no route analysis is recorded for an event then that event is skipped. If none are available, return inf.
        """
        events = self.run(simulation_days, breakdown_rate)
        response_times = []
        for event in events:
            analysis = event.get('route_analysis')
            if analysis is not None and 'travel_time' in analysis:
                response_times.append(analysis['travel_time'])
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            logging.info(f"Average response time computed: {avg_time:.2f} minutes")
            return avg_time
        else:
            logging.warning("No valid route analyses were computed; returning infinity")
            return float('inf')
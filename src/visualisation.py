from manim import *
from network import Network, VoronoiType
from data import GarageData
import numpy as np
from simulator import BreakdownSimulator
from shapely.geometry import Point
import simpy
import random
import json
from datetime import datetime
import logging

class AnimatedResults(MovingCameraScene):
    def __init__(self, results_path="results/simulation_results_20250129_230927_run1.json",
                 start_time=420, end_time=540, **kwargs):
        super().__init__(**kwargs)
        self.results_path = results_path
        self.start_time = start_time
        self.end_time = end_time

    def setup(self):
        with open(self.results_path) as f:
            all_events = json.load(f)
            self.events = [
                event for event in all_events
                if self.start_time <= event['time'] <= self.end_time
            ]
            print(f"Loaded {len(self.events)} events between {self.format_time(self.start_time)} and {self.format_time(self.end_time)}")

            # Print event details
            for event in self.events:
                print(f"\nEvent at time {event['time']}:")
                print(f"Location: {event['location']['coordinates']}")

        # Setup network with traffic Voronoi
        self.garage_data = GarageData()
        self.network = Network(self.garage_data, [
            'Bristol West', 'Bristol South', 'Bristol East', 'Bristol North West',
            'Filton and Bradley Stoke', 'North Somerset', 'Weston-Super-Mare',
            'North East Somerset', 'Bath', 'Kingswood', 'Thornbury and Yate',
            'Stroud', 'Gloucester', 'Forest of Dean', 'Monmouth', 'Newport East',
            'Newport West'
        ])

        # Analyze routes for all events
        for event in self.events:
            # Create point and extract coordinates
            breakdown_point = Point(event['location']['coordinates'])
            # Flip coordinates from (lng, lat) to (lat, lng) for analyse_route
            coords = (breakdown_point.y, breakdown_point.x)  # Flipped order!

            # Get hour of day for traffic analysis
            hour = (event['time'] // 60) % 24

            analysis = self.network.analyse_route(coords, voronoi_type=VoronoiType.TRAFFIC, hour=hour)
            if analysis:
                event['garage_id'] = analysis['garage']['Postcode']
                event['route_analysis'] = analysis
                # print(f"Route analysis for event at time {event['time']}:")
                print(f"  Garage: {event['garage_id']}")
                print(f"  Has route geometry: {'route_geometry' in analysis}")
                if 'route_geometry' in analysis:
                    print(f"  Route geometry type: {type(analysis['route_geometry'])}")
            else:
                event['garage_id'] = None
                event['route_analysis'] = None
                # print(f"No route analysis for event at time {event['time']}")

        print(f"Analyzed routes for {len(self.events)} events")

        # Setup camera and transforms
        boundary = self.network.get_network().geometry.iloc[0]
        self.minx, self.miny, self.maxx, self.maxy = boundary.bounds
        self.scale_factor = 0.005
        self.center_x = (self.minx + self.maxx) / 2
        self.center_y = (self.miny + self.maxy) / 2

        # Set camera frame
        width = (self.maxx - self.minx) / self.scale_factor
        height = (self.maxy - self.miny) / self.scale_factor
        self.camera.frame.set_width(width * 1.1)
        self.camera.frame.set_height(height * 1.1)
        self.camera.frame.move_to(self._transform_point((self.center_x, self.center_y)))
        self.camera.background_color = BLACK

    def format_time(self, minutes):
        """Convert minutes since midnight to HH:MM format"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"

    def construct(self):
        # Draw constituencies
        constituencies = self.create_constituencies()
        self.add(constituencies)

        # Create garage dots with a lookup dictionary
        garage_dots = {}  # Store garage dots by postcode for easy lookup
        garages = VGroup()
        for _, garage in self.network.filtered_garages.iterrows():
            point = self._transform_point((garage.geometry.x, garage.geometry.y))
            dot = Dot(point=point, color=RED, radius=1)
            garages.add(dot)
            garage_dots[garage['Postcode']] = dot
        self.add(garages)

        # Create time display
        time_display = Text(self.format_time(self.start_time), color=WHITE).scale(16)
        time_display.move_to(
            self.camera.frame.get_left() + self.camera.frame.get_bottom() +
            np.array([time_display.width/2 + 5, time_display.height/2 + 5, 0])
        )
        self.add(time_display)

        # Keep track of active breakdowns with their duration and state
        active_breakdowns = {}  # {dot: {start_time, total_duration, route, travel_time, state}}

        # Animate time progression
        for current_time in range(self.start_time, self.end_time + 1):
            # Add new breakdowns for this minute
            new_animations = []
            for event in self.events:
                if event['time'] == current_time:
                    # Calculate times
                    travel_time = event['route_analysis']['travel_time'] if event['route_analysis'] else 0
                    service_time = 30  # 30 min service time
                    total_duration = 2 * travel_time + service_time

                    # Create breakdown dot and route
                    location = event['location']['coordinates']
                    point = self._transform_point((location[0], location[1]))
                    dot = Dot(point=point, color=PURPLE, radius=1)  # Changed from BLUE to PURPLE for traveling out
                    route = self.create_route(event['route_analysis']['route_geometry']) if event['route_analysis'] else None

                    # REORDERED: Create route first, then dot
                    if route:
                        new_animations.append(Create(route))  # Route added first
                    new_animations.append(FadeIn(dot, scale=1.5))  # Dot added after

                    # Store with calculated duration and state info
                    active_breakdowns[dot] = {
                        'start_time': current_time,
                        'total_duration': total_duration,
                        'route': route,
                        'travel_time': travel_time,
                        'service_time': service_time,
                        'state': 'traveling_out'
                    }

                    print(f"\nNew breakdown at {self.format_time(current_time)}:")
                    print(f"Travel time: {travel_time} minutes")
                    print(f"Service time: {service_time} minutes")
                    print(f"Total duration: {total_duration} minutes")

                    # Garage flash remains on top
                    if event['garage_id'] and event['garage_id'] in garage_dots:
                        garage_dot = garage_dots[event['garage_id']]
                        garage_flash = Succession(
                            garage_dot.animate.set_color(GREEN),
                            Wait(0.5),
                            garage_dot.animate.set_color(RED),
                        )
                        new_animations.append(garage_flash)

            # Update states and colors of active breakdowns
            for dot, data in active_breakdowns.items():
                elapsed_time = current_time - data['start_time']

                # Calculate phase timings (in minutes)
                travel_out_end = data['travel_time']
                service_end = travel_out_end + data['service_time']
                travel_back_end = service_end + data['travel_time']

                # Update state and color based on elapsed time
                if elapsed_time < travel_out_end:
                    if data['state'] != 'traveling_out':
                        data['state'] = 'traveling_out'
                        dot.set_color(PURPLE)  # Purple for traveling out
                elif elapsed_time < service_end:
                    if data['state'] != 'servicing':
                        data['state'] = 'servicing'
                        dot.set_color(GREEN)  # Green for servicing
                elif elapsed_time < travel_back_end:
                    if data['state'] != 'traveling_back':
                        data['state'] = 'traveling_back'
                        dot.set_color(TEAL)  # Teal for traveling back

            # Remove completed breakdowns
            fade_out_animations = []
            dots_to_remove = []
            for dot, data in active_breakdowns.items():
                elapsed_time = current_time - data['start_time']
                if elapsed_time >= data['total_duration']:
                    fade_out_animations.append(FadeOut(dot, scale=0.5))
                    if data['route']:
                        fade_out_animations.append(FadeOut(data['route']))
                    dots_to_remove.append(dot)

            for dot in dots_to_remove:
                del active_breakdowns[dot]

            # Update time display
            new_time_display = Text(self.format_time(current_time), color=WHITE).scale(16)
            new_time_display.move_to(time_display.get_center())

            # Play all animations together
            all_animations = [
                *new_animations,
                *fade_out_animations,
                Transform(time_display, new_time_display)
            ]

            if all_animations:
                self.play(*all_animations, run_time=0.25)  # Increased animation time for better visibility
            else:
                self.wait(0.1)

    def create_constituencies(self):
        constituencies = VGroup()
        for name, poly in self.network._constituency_boundaries.items():
            if poly.geom_type == 'Polygon':
                # Handle exterior boundary
                exterior_points = [self._transform_point((x, y)) for x, y in poly.exterior.coords]
                constituency = VMobject(color=GREY, fill_opacity=0.2)
                constituency.set_points_as_corners(exterior_points)

                # Handle interior holes
                for interior in poly.interiors:
                    interior_points = [self._transform_point((x, y)) for x, y in interior.coords]
                    constituency.add_points_as_corners(interior_points)

                constituencies.add(constituency)

            elif poly.geom_type == 'MultiPolygon':
                for sub_poly in poly.geoms:
                    # Handle exterior boundary
                    exterior_points = [self._transform_point((x, y)) for x, y in sub_poly.exterior.coords]
                    constituency = VMobject(color=GREY, fill_opacity=0.2)
                    constituency.set_points_as_corners(exterior_points)

                    # Handle interior holes
                    for interior in sub_poly.interiors:
                        interior_points = [self._transform_point((x, y)) for x, y in interior.coords]
                        constituency.add_points_as_corners(interior_points)

                    constituencies.add(constituency)
        return constituencies

    def create_garages(self):
        # This method is no longer used
        pass

    def _transform_point(self, coord):
        x = (coord[0] - self.center_x) / self.scale_factor
        y = (coord[1] - self.center_y) / self.scale_factor
        return np.array([x, y, 0])

    def create_route(self, geometry):
        """Create a route line from geometry"""
        try:
            # print(f"Creating route from geometry: {geometry}")
            points = [self._transform_point((x, y)) for x, y in geometry.coords]
            print(f"Transformed {len(points)} points")
            route = VMobject()
            route.set_points_smoothly(points)
            route.set_stroke(color=GRAY, width=50, opacity=0.8)
            return route
        except Exception as e:
            print(f"Error creating route: {e}")
            return None

if __name__ == "__main__":
    # For direct execution via Manim CLI
    scene = AnimatedResults(
        results_path="results/simulation_results_20250129_230927_run1.json",
        start_time=420,
        end_time=1080,
    )
    scene.render()

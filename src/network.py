import logging
import os
from typing import List, Tuple
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle
from pathlib import Path

import shapely
import networkx as nx
from data import GarageData
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, MultiPoint, LineString
from shapely.ops import unary_union
import osmnx as ox
import numpy as np
import geovoronoi as gv

from data import TrafficData

class VoronoiType(Enum):
    NONE = "none"
    EUCLIDEAN = "euclidean"
    NETWORK = "network"
    TRAFFIC = "traffic"

class Network:
    def __init__(self, garage_data, constituencies, voronoi_type: VoronoiType = None, traffic_hour: int = None, use_voronoi_cache: bool = True):
        """Initialize network with garage data and constituency list

        Args:
            garage_data: GarageData object containing garage locations
            constituencies: List of constituency names
            voronoi_type: Type of Voronoi regions to calculate
            traffic_hour: Specific hour for traffic Voronoi (7-18). Only used if voronoi_type is TRAFFIC
            use_voronoi_cache: Whether to use cached Voronoi data
        """
        self.garage_data = garage_data
        self.constituencies = sorted(constituencies)
        self._constituency_boundaries = {}
        self.boundary = None
        self.filtered_garages = None
        self.traffic_hour = traffic_hour
        self._road_network = None

        # Get network boundary from constituencies
        network = self.get_network()
        self.boundary = network.geometry.iloc[0]
        self.network = network

        # Filter garages to only those within boundary
        self.filtered_garages = self._filter_garages()

        # Initialize road network if we need it
        if voronoi_type in [VoronoiType.NETWORK, VoronoiType.TRAFFIC] or voronoi_type is None:
            self._road_network = self.get_road_network()

        # Only initialize Voronoi if specifically requested
        if voronoi_type is not None:
            self.cache_key = self._generate_cache_key()
            self.cache_file = Path(f"data/voronoi_cache_{self.cache_key}.pkl")

            self.euclidean_voronoi = {}
            self.network_voronoi = {}
            self.traffic_voronoi = {}

            if use_voronoi_cache and self._load_from_cache():
                logging.info(f"Loaded Voronoi data from cache {self.cache_file}")
            else:
                if voronoi_type == VoronoiType.EUCLIDEAN:
                    self.euclidean_voronoi = self.get_euclidean_voronoi()
                elif voronoi_type == VoronoiType.NETWORK:
                    self.network_voronoi = self.get_network_voronoi()
                elif voronoi_type == VoronoiType.TRAFFIC:
                    self.traffic_voronoi = self.get_traffic_voronoi()

                if use_voronoi_cache:
                    self._save_to_cache()

        self.constituency_populations = self.load_constituency_populations()

    def _filter_garages(self):
        """Filter garages to only those within the network boundary"""
        filtered = self.garage_data.gdf[self.garage_data.gdf.geometry.within(self.boundary)]
        logging.debug(f"Filtered to {len(filtered)} garages within boundary")
        return filtered

    def _init_traffic_voronoi(self):
        """Initialize traffic-weighted Voronoi regions"""
        if not hasattr(self, 'traffic_voronoi'):
            self.traffic_voronoi = {}
        return self.traffic_voronoi

    def _calculate_voronoi_for_hour(self, road_network, filtered_garages, hour):
        """Calculate traffic-weighted Voronoi regions for a specific hour"""
        logging.debug(f"Calculating traffic Voronoi for hour {hour}")

        # Load and apply traffic factors
        traffic_data = TrafficData()
        traffic_factors = traffic_data.traffic_factors

        # Create a copy of the network for this hour's calculations
        G = road_network.copy()

        # Apply traffic factors to network
        for u, v, k, data in G.edges(data=True, keys=True):
            highway = data.get('highway', 'default')
            if isinstance(highway, list):
                highway = highway[0]

            # Get traffic factor for this road type and hour
            road_factors = traffic_factors.get(highway, traffic_factors['default'])
            if isinstance(road_factors, dict):
                traffic_factor = road_factors.get(str(hour), 1.0)
            else:
                traffic_factor = road_factors

            # Apply factor to travel time
            data['traffic_time'] = data['travel_time'] * traffic_factor

        # Convert garage points to network nodes
        garage_nodes = {}
        node_to_garage = {}
        for idx, garage in filtered_garages.iterrows():
            nearest_node = ox.nearest_nodes(
                G,
                garage.geometry.x,
                garage.geometry.y
            )
            garage_nodes[idx] = nearest_node
            node_to_garage[nearest_node] = idx

        # Calculate Voronoi cells using traffic-weighted network
        try:
            assigner = nx.algorithms.voronoi.voronoi_cells(
                G,
                list(garage_nodes.values()),
                weight='traffic_time'
            )
        except nx.NetworkXUnfeasible:
            logging.error(f"Voronoi cells calculation failed for hour {hour}")
            return {}

        # Create region polygons from edge assignments
        voronoi_regions = {}
        for node_id, cell_nodes in assigner.items():
            if node_id == "unreachable":
                continue

            # Get original garage index
            garage_id = node_to_garage.get(node_id)
            if garage_id is None:
                continue

            # Get edges that are entirely within the cell
            cell_edges = []
            for u, v, data in G.edges(data=True):
                if u in cell_nodes and v in cell_nodes:
                    if 'geometry' in data:
                        cell_edges.append(data['geometry'])
                    else:
                        u_data = G.nodes[u]
                        v_data = G.nodes[v]
                        cell_edges.append(LineString([
                            (u_data['x'], u_data['y']),
                            (v_data['x'], v_data['y'])
                        ]))

            # Create polygon from edge network
            if cell_edges:
                try:
                    buffer_distance = 0.0001
                    buffered_lines = [line.buffer(buffer_distance) for line in cell_edges]
                    merged = unary_union(buffered_lines)

                    if not merged.is_valid:
                        merged = merged.buffer(0)

                    polygon = merged.simplify(buffer_distance)

                    if polygon.intersects(self.boundary):
                        region = polygon.intersection(self.boundary)
                        if region.is_valid and not region.is_empty:
                            voronoi_regions[garage_id] = region

                except Exception as e:
                    logging.error(f"Error creating region for garage {garage_id} at hour {hour}: {e}")

        return voronoi_regions

    def load_geojson(self):
        logging.debug("Loading England and Wales WPC data")
        england_wpc = gpd.read_file("data/England_WPC.json")
        wales_wpc = gpd.read_file("data/Wales_WPC.json")
        NI_wpc = gpd.read_file("data/NI_WPC.json")
        Scotland_wpc = gpd.read_file("data/Scotland_WPC.json")

        combined_wpc = pd.concat([england_wpc, wales_wpc, NI_wpc, Scotland_wpc], ignore_index=True)
        logging.debug(f"Combined dataset has {len(combined_wpc)} rows")

        logging.debug("Filtering for Bristol region constituencies")
        focused_region = combined_wpc[combined_wpc['PCON13NM'].isin(self.constituencies)].copy()
        logging.info(f"Found {len(focused_region)} constituencies in the region")

        # Store individual constituency boundaries
        for _, constituency in focused_region.iterrows():
            name = constituency['PCON13NM']
            geometry = constituency.geometry.buffer(0)  # Clean geometry
            if geometry.is_valid:
                self._constituency_boundaries[name] = geometry
            else:
                logging.warning(f"Invalid geometry for constituency {name}")

        logging.debug("Cleaning geometry and creating boundary")
        focused_region.geometry = focused_region.geometry.buffer(0)
        focused_region = focused_region[focused_region.geometry.is_valid]
        boundary_shape = shapely.ops.unary_union(focused_region.geometry)

        if isinstance(boundary_shape, MultiPolygon):
            logging.debug("Converting MultiPolygon to unified boundary")
            boundary_shape = shapely.ops.unary_union(boundary_shape)

        logging.debug("Boundary shape created successfully")
        return boundary_shape

    def get_constituency_boundary(self, constituency_name: str) -> shapely.Polygon:
        """Get the boundary polygon for a specific constituency"""
        if constituency_name not in self._constituency_boundaries:
            logging.warning(f"Constituency {constituency_name} not found in boundaries")
            return None
        return self._constituency_boundaries[constituency_name]

    def get_network(self):
        logging.debug("Getting network from GeoJSON")
        geojson = self.load_geojson()
        self.boundary = geojson
        gdf = gpd.GeoDataFrame(geometry=[geojson])
        logging.debug("Created GeoDataFrame from boundary shape")
        return gdf

    def get_road_network(self):
        """Get road network from cache or download from OSM"""
        # Return cached network if available
        if hasattr(self, '_road_network') and self._road_network is not None:
            return self._road_network

        ROAD_SPEEDS = {
            'motorway': 70,  # mph
            'trunk': 50,
            'primary': 40,
            'secondary': 30,
            'tertiary': 30,
            'residential': 20,
            'unclassified': 20,
            'default': 30
        }

        cache_path = "data/road_network.graphml"
        if os.path.exists(cache_path):
            logging.debug("Loading road network from cache")
            road_network = ox.load_graphml(cache_path)
        else:
            logging.debug("Downloading road network from OSM")
            road_network = ox.graph_from_polygon(self.boundary, network_type='drive')
            logging.debug("Caching road network to file")
            ox.save_graphml(road_network, cache_path)

        # Add travel time to edges based on road type speeds
        for u, v, k, data in road_network.edges(data=True, keys=True):
            if 'length' in data:
                highway = data.get('highway', 'default')
                if isinstance(highway, list):
                    highway = highway[0]
                speed = ROAD_SPEEDS.get(highway, ROAD_SPEEDS['default'])
                speed_ms = speed * 0.44704  # Convert mph to m/s
                data['travel_time'] = data['length'] / speed_ms
            else:
                # If length not available, calculate from coordinates
                start = road_network.nodes[u]
                end = road_network.nodes[v]
                dist = ((start['x'] - end['x'])**2 + (start['y'] - end['y'])**2)**0.5
                data['length'] = dist
                speed = ROAD_SPEEDS['default']
                speed_ms = speed * 0.44704
                data['travel_time'] = dist / speed_ms

        self._road_network = road_network
        return road_network

    def get_euclidean_voronoi(self):
        """Calculate Euclidean Voronoi regions for garage locations"""
        logging.debug("Calculating Euclidean Voronoi regions")

        # Use stored filtered garages
        if len(self.filtered_garages) == 0:
            logging.warning("No garages found within boundary")
            return None

        # Collect garage points with their original indices
        garage_data = []
        for idx, row in self.filtered_garages.iterrows():
            garage_data.append({
                'id': idx,
                'geometry': row.geometry
            })

        # Extract coordinates as numpy array
        points = np.array([[g['geometry'].x, g['geometry'].y] for g in garage_data])

        # Calculate Voronoi regions
        poly_shapes, pts = gv.voronoi_regions_from_coords(points, self.boundary)

        # Match regions to garages using spatial containment
        voronoi_regions = {}
        for region_id, poly in poly_shapes.items():
            # Find which garage point is contained in this region
            match_found = False
            for garage in garage_data:
                if poly.contains(garage['geometry']):
                    voronoi_regions[garage['id']] = poly
                    match_found = True
                    break

            if not match_found:
                # Fallback: find nearest garage point
                min_dinstance = float('inf')
                closest_garage = None
                for garage in garage_data:
                    dist = poly.distance(garage['geometry'])
                    if dist < min_dinstance:
                        min_dinstance = dist
                        closest_garage = garage

                if closest_garage:
                    logging.warning(f"Region {region_id} using fallback to closest garage (distance: {min_dinstance})")
                    voronoi_regions[closest_garage['id']] = poly

        logging.info(f"Created {len(voronoi_regions)} Euclidean Voronoi regions")
        return voronoi_regions

    def get_network_voronoi(self):
        """Calculate non-overlapping network Voronoi regions using edge buffers"""
        logging.debug("Calculating precise network Voronoi regions")

        # Get garage points within boundary
        filtered_garages = self.garage_data.gdf[self.garage_data.gdf.geometry.within(self.network.geometry.iloc[0])]

        if len(filtered_garages) == 0:
            logging.warning("No garages found within boundary")
            return None

        # Get the road network with travel times
        road_network = self.get_road_network()

        # Convert garage points to network nodes with bidirectional mapping
        garage_nodes = {}
        node_to_garage = {}  # NEW: Map from OSM node IDs to garage indices
        for idx, garage in filtered_garages.iterrows():
            # Find nearest network node to garage
            nearest_node = ox.nearest_nodes(
                road_network,
                garage.geometry.x,
                garage.geometry.y
            )
            garage_nodes[idx] = nearest_node
            node_to_garage[nearest_node] = idx  # NEW: Store reverse mapping

        # Calculate shortest paths from all garages simultaneously
        logging.debug("Calculating all-pairs shortest paths")
        assigner = nx.algorithms.voronoi.voronoi_cells(
            road_network,
            garage_nodes.values(),
            weight='travel_time'
        )

        # Create region polygons from edge assignments
        voronoi_regions = {}
        for garage_node_id, cell_nodes in assigner.items():
            # NEW: Convert OSM node ID back to original garage index
            garage_id = node_to_garage.get(garage_node_id)
            if garage_id is None:
                logging.warning(f"No garage found for node {garage_node_id}")
                continue

            # Get edges that are entirely within the cell (both nodes in cell)
            cell_edges = []
            for u, v, data in road_network.edges(data=True):
                if u in cell_nodes and v in cell_nodes:
                    if 'geometry' in data:
                        # Use the actual road geometry if available
                        cell_edges.append(data['geometry'])
                    else:
                        # Create straight line if no geometry
                        u_data = road_network.nodes[u]
                        v_data = road_network.nodes[v]
                        cell_edges.append(LineString([
                            (u_data['x'], u_data['y']),
                            (v_data['x'], v_data['y'])
                        ]))

            # Create polygon from edge network
            if cell_edges:
                try:
                    # Create valid polygon
                    buffer_distance = 0.0001  # Adjust based on coordinate system
                    buffered_lines = [line.buffer(buffer_distance) for line in cell_edges]
                    merged = unary_union(buffered_lines)

                    # Fix invalid geometries
                    if not merged.is_valid:
                        merged = merged.buffer(0)

                    # Simplify the merged polygon to remove artifacts
                    polygon = merged.simplify(buffer_distance)

                    if polygon.intersects(self.boundary):
                        region = polygon.intersection(self.boundary)
                        if region.is_valid and not region.is_empty:
                            voronoi_regions[garage_id] = region  # Use original garage index

                except Exception as e:
                    logging.error(f"Error creating region for garage {garage_id}: {e}")

        logging.info(f"Created {len(voronoi_regions)} non-overlapping network Voronoi regions")
        return voronoi_regions

    def get_traffic_voronoi(self):
        """Calculate traffic-weighted Voronoi regions for specific hour"""
        logging.debug("Calculating traffic Voronoi regions")
        if not hasattr(self, '_road_network'):
            self._road_network = self.get_road_network()

        if not self.traffic_hour or not 7 <= self.traffic_hour <= 18:
            logging.warning(f"Invalid traffic hour {self.traffic_hour}, using default hour 8")
            self.traffic_hour = 8

        traffic_voronoi = {}
        try:
            regions = self._calculate_voronoi_for_hour(
                self._road_network,
                self.filtered_garages,
                self.traffic_hour
            )
            traffic_voronoi[self.traffic_hour] = regions
            logging.debug(f"Calculated traffic Voronoi for hour {self.traffic_hour}")
        except Exception as e:
            logging.error(f"Failed to calculate traffic Voronoi for hour {self.traffic_hour}: {e}")

        return traffic_voronoi

    def show_network(
            self,
            show_garages=False,
            show_roads=False,
            show_constituencies=False,
            show_traffic=False,
            voronoi_type: VoronoiType = None,
            traffic_hour: int = None,
            coord: Tuple[float, float] = None,
            show_garage_labels=False,
            ):
        """
        Plot network visualisation with optional layers

        Args:
            show_garages (bool): Whether to show garage locations
            show_roads (bool): Whether to show the road network
            show_constituencies (bool): Whether to show constituency boundaries
            show_traffic (bool): Whether to show the traffic data
            voronoi_type (VoronoiType): Type of Voronoi diagram to display (if any)
            show_garage_labels (bool): Whether to show garage postcode labels
        """
        logging.debug("Plotting network visualisation")

        # Create the base plot
        ax = self.network.plot(edgecolor='black', facecolor='none')

        if show_constituencies:
            logging.debug("Adding constituency boundaries to plot")
            # Use stored constituency boundaries instead of reloading
            for name, boundary in self._constituency_boundaries.items():
                constituency_gdf = gpd.GeoDataFrame(geometry=[boundary])
                constituency_gdf.plot(
                    ax=ax,
                    facecolor='none',  # No fill
                    edgecolor='black',  # Blue border
                    linewidth=2,       # Thicker line
                    label=name
                )

                # Add constituency name label at centroid
                # centroid = boundary.centroid
                # plt.annotate(
                #     name,
                #     xy=(centroid.x, centroid.y),
                #     fontsize=6,
                #     alpha=1.0,
                #     bbox=dict(
                #         facecolor='white',
                #         alpha=0.7,
                #         edgecolor='none',
                #         pad=0.5
                #     )
                # )

        # Store region colors for garage markers
        garage_colors = {}
        filtered_garages = self.garage_data.gdf[self.garage_data.gdf.geometry.within(self.network.geometry.iloc[0])]

        if voronoi_type:
            logging.debug(f"Adding {voronoi_type.value} Voronoi regions to plot")

            if voronoi_type == VoronoiType.NONE:
                pass
            elif voronoi_type == VoronoiType.EUCLIDEAN and self.euclidean_voronoi:
                for idx, (garage_id, region) in enumerate(self.euclidean_voronoi.items()):
                    if isinstance(region, (shapely.Polygon, shapely.MultiPolygon)):
                        color = plt.cm.tab20(idx / len(filtered_garages))
                        garage_colors[garage_id] = color
                        gpd.GeoDataFrame(geometry=[region]).plot(
                            ax=ax,
                            color=color,
                            alpha=0.7,
                            edgecolor=color,
                            # linewidth=1  # Added thicker edges
                        )
            elif voronoi_type == VoronoiType.NETWORK and self.network_voronoi:
                for idx, (garage_id, region) in enumerate(self.network_voronoi.items()):
                    if isinstance(region, (shapely.Polygon, shapely.MultiPolygon)):
                        color = plt.cm.tab20(idx / len(filtered_garages))
                        garage_colors[garage_id] = color
                        gpd.GeoDataFrame(geometry=[region]).plot(
                            ax=ax,
                            color=color,
                            alpha=1.0,
                            edgecolor=color
                        )
            elif voronoi_type == VoronoiType.TRAFFIC:
                logging.debug(f"Adding traffic-weighted Voronoi regions for hour {traffic_hour}")
                if hasattr(self, 'traffic_voronoi') and self.traffic_voronoi and (traffic_hour in self.traffic_voronoi):
                    for idx, (garage_id, region) in enumerate(self.traffic_voronoi[traffic_hour].items()):
                        if isinstance(region, (shapely.Polygon, shapely.MultiPolygon)):
                            color = plt.cm.tab20(idx / len(filtered_garages))
                            garage_colors[garage_id] = color
                            gpd.GeoDataFrame(geometry=[region]).plot(
                                ax=ax,
                                color=color,
                                alpha=0.7,
                                edgecolor=color,
                                linewidth=1
                            )
                else:
                    logging.debug("No traffic-weighted Voronoi regions available; skipping traffic Voronoi layer.")

        if show_roads:
            logging.debug("Adding road network to plot")
            roads_gdf = gpd.GeoDataFrame(geometry=[
                data['geometry'] for _, _, data in self.get_road_network().edges(data=True)
                if 'geometry' in data
            ])

            roads_gdf.plot(ax=ax, color='black', linewidth=0.5, alpha=0.3)

            if show_traffic:
                logging.debug("Adding traffic data visualisation")
                traffic_df = TrafficData().df
                # TODO: Add traffic data visualiation

        if show_garages:
            logging.debug("Adding garage locations to plot")
            # Plot garages with matching region colors
            for idx, garage in filtered_garages.iterrows():
                color = garage_colors.get(idx, 'red')  # Default to red if no region color found
                plt.scatter(
                    garage.geometry.x,
                    garage.geometry.y,
                    color=color,
                    edgecolor='black',
                    s=200,
                    zorder=3,
                    marker='X'
                )

                if show_garage_labels:
                    plt.annotate(
                        garage['Postcode'],
                        xy=(garage.geometry.x, garage.geometry.y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(
                            facecolor='white',
                            alpha=0.7,
                            edgecolor='none',
                            pad=0.5
                        )
                    )

        if coord:
            logging.debug(f"Adding coordinate {coord} to plot")
            # Find and plot containing garage and route
            route_analysis = self.analyse_route(
                coord,
                voronoi_type if voronoi_type else VoronoiType.NETWORK,
                traffic_hour
            )

            # Plot coordinate point
            plt.scatter(
                coord[1],  # longitude (x)
                coord[0],  # latitude (y)
                color='blue',
                s=40,
                zorder=4,
                marker='x',
                label='Selected Location'
            )

            # Plot route if found
            if route_analysis:
                # Convert LineString to coordinate pairs
                x, y = route_analysis['route_geometry'].xy
                plt.plot(
                    x, y,
                    color='purple',
                    linewidth=2,
                    linestyle='--',
                    zorder=3,
                    label=f"Route ({route_analysis['travel_time']} mins)"
                )

                # Plot containing garage
                plt.scatter(
                    route_analysis['garage']['geometry'].x,
                    route_analysis['garage']['geometry'].y,
                    color='red',
                    edgecolor='black',
                    s=150,
                    zorder=5,
                    marker='*',
                    label=f"Responsible Garage ({route_analysis['garage'].get('Postcode', 'Unknown')})"
                )
            else:
                logging.warning("No route found to plot")

        plt.gcf().set_size_inches(12, 8)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f'plots/network_visualisation_{voronoi_type.value}_{traffic_hour}.png', dpi=300)
        plt.show()
        logging.debug("Network visualisation complete")

        # Adjust legend to account for route
        if coord:
            ax.legend(loc='upper right')

    def find_garage_for_coordinate(self, coord: Tuple[float, float], voronoi_type: VoronoiType, hour: int = None) -> dict:
        """Find which garage's Voronoi region contains the given coordinate"""
        # Convert to Shapely point (lng, lat)
        original_point = shapely.Point(coord[1], coord[0])

        # First check if point is within boundary
        if not self.boundary.contains(original_point):
            logging.warning(f"Coordinate {coord} is outside the network boundary")
            return None

        # Get road network and project point onto nearest edge
        G = self.get_road_network()
        edge = ox.distance.nearest_edges(G, original_point.x, original_point.y)

        # Get the geometry of the nearest edge
        edge_data = G.edges[edge[0], edge[1], edge[2]]
        if 'geometry' in edge_data:
            edge_line = edge_data['geometry']
        else:
            # Create straight line between nodes if no geometry
            u_pt = shapely.Point(G.nodes[edge[0]]['x'], G.nodes[edge[0]]['y'])
            v_pt = shapely.Point(G.nodes[edge[1]]['x'], G.nodes[edge[1]]['y'])
            edge_line = shapely.LineString([u_pt, v_pt])

        # Project point onto the edge line
        projected_point = edge_line.interpolate(edge_line.project(original_point))

        logging.debug(f"Original point: {original_point.wkt}")
        logging.debug(f"Projected point onto network: {projected_point.wkt}")

        # Now use the projected point for Voronoi checks
        point = projected_point

        # Get and validate the appropriate Voronoi regions
        if voronoi_type == VoronoiType.EUCLIDEAN:
            regions = self.euclidean_voronoi
            logging.debug(f"Using Euclidean Voronoi with {len(regions) if regions else 0} regions")
        elif voronoi_type == VoronoiType.NETWORK:
            regions = self.network_voronoi
            logging.debug(f"Using Network Voronoi with {len(regions) if regions else 0} regions")
        elif voronoi_type == VoronoiType.TRAFFIC:
            if hour is None:
                logging.warning("No hour specified for traffic Voronoi")
                return None
            regions = self.traffic_voronoi.get(hour, {})
            logging.debug(f"Using Traffic Voronoi for hour {hour} with {len(regions)} regions")
        else:
            logging.warning(f"Invalid Voronoi type: {voronoi_type}")
            return None

        if not regions:
            logging.warning(f"No Voronoi regions available for type {voronoi_type}")
            return None

        # Find containing region
        containing_regions = []
        for garage_id, region in regions.items():
            # Validate region geometry
            if not isinstance(region, (shapely.Polygon, shapely.MultiPolygon)):
                logging.warning(f"Invalid region geometry type for garage {garage_id}: {type(region)}")
                continue
            if not region.is_valid:
                logging.warning(f"Invalid region geometry for garage {garage_id}")
                continue

            # Check if point is in region with a small buffer for numerical precision
            try:
                if region.buffer(0.0001).contains(point):  # Add small buffer for edge cases
                    containing_regions.append(garage_id)
            except Exception as e:
                logging.error(f"Error checking if region contains point for garage {garage_id}: {e}")
                continue

        if not containing_regions:
            logging.warning(f"Coordinate {coord} not found in any {voronoi_type.value} Voronoi region")
            # Debug: check if point is near any region boundary
            for garage_id, region in regions.items():
                if region.distance(point) < 0.0001:  # Small threshold
                    logging.debug(f"Point is very close to region {garage_id} (distance: {region.distance(point)})")
            return None

        if len(containing_regions) > 1:
            logging.warning(f"Coordinate {coord} found in multiple regions: {containing_regions}")

        # Get the garage data using the first containing region
        try:
            garage_id = containing_regions[0]
            if garage_id not in self.filtered_garages.index:  # Use stored filtered garages
                logging.error(f"Garage ID {garage_id} not found in filtered garages (indices: {self.filtered_garages.index.tolist()})")
                return None

            garage_data = self.filtered_garages.loc[garage_id].to_dict()
            logging.debug(f"Found responsible garage: {garage_data.get('Name', garage_id)}")
            return garage_data
        except Exception as e:
            logging.error(f"Error retrieving garage data for ID {garage_id}: {e}")
            return None

    def analyse_route(self, coord: Tuple[float, float], voronoi_type: VoronoiType = None, hour: int = None, cached_network = None) -> dict:
        """Analyze route from coordinate to nearest garage

        Args:
            coord: (latitude, longitude) tuple
            voronoi_type: Not used anymore, kept for backwards compatibility
            hour: Hour of day for traffic weighting (7-18)
            cached_network: Pre-weighted network graph (optional)

        Returns:
            dict: Analysis results including garage info and route metrics
        """
        # Find nearest garage using direct route calculation
        analysis = self.find_nearest_garage(coord, hour)

        if analysis:
            return {
                'garage': analysis['garage'],
                'travel_time': analysis['travel_time'],
                'distance': analysis['distance'],
                'route_geometry': analysis['route_geometry']
            }

        logging.warning(f"Could not find route for coordinate {coord}")
        return None

    def load_constituency_populations(self):
        """Load population data from CSV"""
        pop_df = pd.read_csv("data/constituency_populations.csv")
        return dict(zip(pop_df['constituency'], pop_df['population']))

    def _generate_cache_key(self) -> str:
        """Generate unique hash key for current configuration"""
        # Convert geometry series to individual WKB bytes and concatenate
        geometry_bytes = b''.join(self.filtered_garages.geometry.apply(lambda g: g.wkb).tolist())

        hash_data = {
            'constituencies': self.constituencies,
            'garage_count': len(self.filtered_garages),
            'garage_hash': hashlib.sha256(geometry_bytes).hexdigest()[:16],
            'version': 1  # Increment if cache format changes
        }
        return hashlib.sha256(str(hash_data).encode()).hexdigest()[:16]

    def _load_from_cache(self) -> bool:
        """Attempt to load Voronoi data from cache"""
        if not self.cache_file.exists():
            logging.info("No cache found, calculating fresh")
            return False

        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.euclidean_voronoi = cache_data['euclidean']
                self.network_voronoi = cache_data['network']
                self.traffic_voronoi = cache_data['traffic']
                logging.info(f"Loaded Voronoi data from cache {self.cache_file}")
                return True
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}, recalculating")
            return False

    def _save_to_cache(self):
        """Save current Voronoi data to cache"""
        try:
            cache_data = {
                'euclidean': self.euclidean_voronoi,
                'network': self.network_voronoi,
                'traffic': self.traffic_voronoi
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logging.info(f"Saved Voronoi data to cache {self.cache_file}")
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")

    def find_nearest_garage(self, coord: Tuple[float, float], hour: int = None) -> dict:
        """Find nearest garage by traffic-weighted route time

        Args:
            coord: (latitude, longitude) of breakdown
            hour: Hour of day for traffic weighting (7-18)

        Returns:
            dict: Garage data and route metrics for nearest garage
        """
        if not self._road_network:
            self._road_network = self.get_road_network()

        # Create traffic-weighted network if hour is specified
        if hour and 7 <= hour <= 18:
            G = self._road_network.copy()
            traffic_data = TrafficData()
            traffic_factors = traffic_data.traffic_factors

            # Apply traffic factors to network
            for u, v, k, data in G.edges(data=True, keys=True):
                highway = data.get('highway', 'default')
                if isinstance(highway, list):
                    highway = highway[0]

                # Get traffic factor for this road type and hour
                road_factors = traffic_factors.get(highway, traffic_factors['default'])
                if isinstance(road_factors, dict):
                    traffic_factor = road_factors.get(str(hour), 1.0)
                else:
                    traffic_factor = road_factors

                # Apply factor to travel time
                data['traffic_time'] = data['travel_time'] * traffic_factor
        else:
            G = self._road_network

        best_garage = None
        best_metrics = None
        min_time = float('inf')

        for _, garage in self.filtered_garages.iterrows():
            metrics = self._calculate_route_metrics(
                coord,
                (garage.geometry.y, garage.geometry.x),
                hour,
                cached_network=G
            )

            if metrics and metrics['time'] < min_time:
                min_time = metrics['time']
                best_metrics = metrics
                best_garage = garage.to_dict()

        if best_garage:
            return {
                'garage': best_garage,
                'travel_time': best_metrics['time'],
                'distance': best_metrics['distance'],
                'route_geometry': best_metrics['route_geometry']
            }
        return None

    def _calculate_route_metrics(self, origin: Tuple[float, float], destination: Tuple[float, float],
                               hour: int = None, cached_network = None) -> dict:
        """Calculate route metrics between two points

        Args:
            origin: (latitude, longitude) of start point
            destination: (latitude, longitude) of end point
            hour: Hour of day for traffic weighting (7-18)
            cached_network: Pre-weighted network graph (optional)

        Returns:
            dict: Route metrics including time, distance, and geometry
        """
        try:
            # Use cached network if provided, otherwise get fresh
            G = cached_network if cached_network is not None else self.get_road_network()

            # Get nearest nodes
            orig_node = ox.nearest_nodes(G, origin[1], origin[0])
            dest_node = ox.nearest_nodes(G, destination[1], destination[0])

            # Calculate shortest path
            weight = 'traffic_time' if hour and 7 <= hour <= 18 else 'travel_time'
            path = nx.shortest_path(G, orig_node, dest_node, weight=weight)

            # Calculate metrics
            distance = 0
            time = 0
            coords = []

            # Collect path coordinates and calculate metrics
            for u, v in zip(path[:-1], path[1:]):
                edge_data = G.edges[u, v, 0]

                # Add distance
                if 'length' in edge_data:
                    distance += edge_data['length']

                # Add time (use traffic_time if available, otherwise travel_time)
                time += edge_data.get(weight, edge_data.get('travel_time', 0))

                # Collect coordinates for route geometry
                if 'geometry' in edge_data:
                    coords.extend(list(edge_data['geometry'].coords))
                else:
                    coords.extend([(G.nodes[u]['x'], G.nodes[u]['y']),
                                 (G.nodes[v]['x'], G.nodes[v]['y'])])

            # Create route geometry
            route_geom = LineString(coords)

            return {
                'time': time / 60,  # Convert to minutes
                'distance': distance,
                'route_geometry': route_geom
            }

        except Exception as e:
            logging.error(f"Error calculating route metrics: {e}")
            return None
import logging
import os
from typing import List, Tuple
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    def __init__(self, garages: GarageData, constituencies: List[str]):
        self.garages = garages
        self.constituencies = constituencies
        self.boundary_shape = None
        self.network = self.get_network()
        # Cache the road network with travel times
        self._road_network = None
        
        # Only initialize Voronoi regions if needed
        self.euclidean_voronoi = None
        self.network_voronoi = None
        self.traffic_voronoi = {}

    def _ensure_voronoi_initialized(self, voronoi_type: VoronoiType):
        """Lazy initialization of Voronoi regions when needed"""
        if voronoi_type == VoronoiType.EUCLIDEAN and self.euclidean_voronoi is None:
            self.euclidean_voronoi = self.get_euclidean_voronoi()
        elif voronoi_type == VoronoiType.NETWORK and self.network_voronoi is None:
            self.network_voronoi = self.get_network_voronoi()
        elif voronoi_type == VoronoiType.TRAFFIC and not self.traffic_voronoi:
            self._init_traffic_voronoi()

    def _init_traffic_voronoi(self):
        """Optimised traffic Voronoi initialisation with precomputed factors"""
        logging.debug("Optimised traffic Voronoi initialisation")
        valid_hours = range(7, 19)

        # Use cached road network
        road_network = self.get_road_network()

        # Get filtered garages once
        filtered_garages = self.garages.gdf[self.garages.gdf.geometry.within(self.network.geometry.iloc[0])]
        if len(filtered_garages) == 0:
            logging.warning("No garages found within boundary")
            return

        # Convert garage points to network nodes once
        garage_nodes = {}
        for idx, garage in filtered_garages.iterrows():
            nearest_node = ox.nearest_nodes(
                road_network,
                garage.geometry.x,
                garage.geometry.y
            )
            garage_nodes[idx] = nearest_node

        # Precompute traffic factors for all hours and road types
        logging.debug("Precomputing traffic factors")
        traffic_data = TrafficData()
        traffic_factors = traffic_data.traffic_factors

        # Create networks with precomputed weights for each hour
        weighted_networks = {}
        for hour in valid_hours:
            G_hour = road_network.copy()
            for u, v, k, data in G_hour.edges(data=True, keys=True):
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
            weighted_networks[hour] = G_hour

        logging.debug("Calculating Voronoi regions in parallel")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit all calculations to thread pool
            future_to_hour = {
                executor.submit(
                    self._calculate_voronoi_for_hour,
                    weighted_networks[hour],
                    garage_nodes
                ): hour
                for hour in valid_hours
            }

            # Collect results as they complete
            for future in as_completed(future_to_hour):
                hour = future_to_hour[future]
                try:
                    regions = future.result()
                    if regions:
                        self.traffic_voronoi[hour] = regions
                        logging.debug(f"Completed traffic Voronoi for hour {hour}")
                except Exception as e:
                    logging.error(f"Error calculating traffic Voronoi for hour {hour}: {e}")

        logging.info(f"Initialised traffic Voronoi regions for {len(self.traffic_voronoi)} hours")

    def _calculate_voronoi_for_hour(self, G, garage_nodes):
        """Calculate Voronoi regions for a pre-weighted network"""
        # Calculate Voronoi cells using pre-weighted network
        assigner = nx.algorithms.voronoi.voronoi_cells(
            G,
            garage_nodes.values(),
            weight='traffic_time'
        )

        # Create region polygons from edge assignments
        voronoi_regions = {}
        for garage_id, cell_nodes in assigner.items():
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
                buffer_distance = 0.0001
                buffered_lines = [line.buffer(buffer_distance) for line in cell_edges]
                merged = unary_union(buffered_lines)
                polygon = merged.simplify(buffer_distance)

                if polygon.intersects(self.boundary_shape):
                    region = polygon.intersection(self.boundary_shape)
                    if not region.is_empty:
                        voronoi_regions[garage_id] = region

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

        logging.debug("Cleaning geometry and creating boundary")
        focused_region.geometry = focused_region.geometry.buffer(0)
        focused_region = focused_region[focused_region.geometry.is_valid]
        boundary_shape = shapely.ops.unary_union(focused_region.geometry)

        if isinstance(boundary_shape, MultiPolygon):
            logging.debug("Converting MultiPolygon to unified boundary")
            boundary_shape = shapely.ops.unary_union(boundary_shape)

        logging.debug("Boundary shape created successfully")
        return boundary_shape

    def get_network(self):
        logging.debug("Getting network from GeoJSON")
        geojson = self.load_geojson()
        self.boundary_shape = geojson
        gdf = gpd.GeoDataFrame(geometry=[geojson])
        logging.debug("Created GeoDataFrame from boundary shape")
        return gdf

    def get_road_network(self):
        """Get cached road network or create new one with travel times"""
        if self._road_network is not None:
            logging.debug("Using cached road network")
            return self._road_network

        logging.debug("Getting road network with travel times")
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

        logging.debug("Creating network graph from boundary polygon")
        cache_path = "data/bristol_network.graphml"

        if os.path.exists(cache_path):
            logging.debug("Loading cached road network from file")
            road_network = ox.load_graphml(cache_path)
        else:
            logging.debug("Downloading road network from OSM")
            road_network = ox.graph_from_polygon(self.boundary_shape, network_type='drive')
            logging.debug("Caching road network to file")
            ox.save_graphml(road_network, cache_path)

        logging.info(f"Created graph with {len(road_network.nodes)} nodes and {len(road_network.edges)} edges")

        # Calculate travel times once
        edge_count = 0
        for u, v, k, data in road_network.edges(data=True, keys=True):
            if 'travel_time' not in data:  # Only calculate if not already present
                highway = data.get('highway', 'default')
                if isinstance(highway, list):
                    highway = highway[0]
                speed = ROAD_SPEEDS.get(highway, ROAD_SPEEDS['default'])
                speed_ms = speed * 0.44704  # Convert mph to m/s

                if 'length' in data:
                    data['travel_time'] = data['length'] / speed_ms
                else:
                    if 'geometry' in data:
                        data['length'] = data['geometry'].length
                        data['travel_time'] = data['length'] / speed_ms
                    else:
                        start = road_network.nodes[u]
                        end = road_network.nodes[v]
                        dist = ((start['x'] - end['x'])**2 + (start['y'] - end['y'])**2)**0.5
                        data['length'] = dist
                        data['travel_time'] = dist / speed_ms
                edge_count += 1
                if edge_count % 10000 == 0:
                    logging.debug(f"Processed {edge_count} edges")

        logging.info(f"Completed travel time calculations for {edge_count} edges")

        # Cache the network
        self._road_network = road_network
        return road_network

    def get_euclidean_voronoi(self):
        """Calculate Euclidean Voronoi regions for garage locations"""
        logging.debug("Calculating Euclidean Voronoi regions")

        # Get garage points within boundary
        filtered_garages = self.garages.gdf[self.garages.gdf.geometry.within(self.network.geometry.iloc[0])]

        # Extract coordinates for Voronoi calculation
        points = np.column_stack([
            filtered_garages.geometry.x,
            filtered_garages.geometry.y
        ])

        if len(points) == 0:
            logging.warning("No garages found within boundary")
            return None

        # Calculate Voronoi regions
        poly_shapes, pts = gv.voronoi_regions_from_coords(points, self.boundary_shape)
        logging.info(f"Created {len(poly_shapes)} Euclidean Voronoi regions")

        return poly_shapes

    def get_network_voronoi(self):
        """Calculate non-overlapping network Voronoi regions using edge buffers"""
        logging.debug("Calculating precise network Voronoi regions")

        # Get garage points within boundary
        filtered_garages = self.garages.gdf[self.garages.gdf.geometry.within(self.network.geometry.iloc[0])]

        if len(filtered_garages) == 0:
            logging.warning("No garages found within boundary")
            return None

        # Get the road network with travel times
        road_network = self.get_road_network()

        # Convert garage points to network nodes
        garage_nodes = {}
        for idx, garage in filtered_garages.iterrows():
            # Find nearest network node to garage
            nearest_node = ox.nearest_nodes(
                road_network,
                garage.geometry.x,
                garage.geometry.y
            )
            garage_nodes[idx] = nearest_node

        # Calculate shortest paths from all garages simultaneously
        logging.debug("Calculating all-pairs shortest paths")
        assigner = nx.algorithms.voronoi.voronoi_cells(
            road_network,
            garage_nodes.values(),
            weight='travel_time'
        )

        # Create region polygons from edge assignments
        voronoi_regions = {}
        for garage_id, cell_nodes in assigner.items():
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
                # Buffer each line slightly and union them
                buffer_distance = 0.0001  # Adjust based on coordinate system
                buffered_lines = [line.buffer(buffer_distance) for line in cell_edges]
                merged = unary_union(buffered_lines)

                # Simplify the merged polygon to remove artifacts
                polygon = merged.simplify(buffer_distance)

                if polygon.intersects(self.boundary_shape):
                    region = polygon.intersection(self.boundary_shape)
                    if not region.is_empty:
                        voronoi_regions[garage_id] = region

        logging.info(f"Created {len(voronoi_regions)} non-overlapping network Voronoi regions")
        return voronoi_regions

    def get_traffic_voronoi(self, hour: int = None):
        """Calculate traffic-weighted Voronoi regions using traffic factors for a given hour"""
        if not 7 <= hour <= 18:
            logging.warning(f"Hour {hour} outside valid range (7-18), using default weights")
            return None

        logging.debug(f"Calculating traffic-weighted Voronoi regions for hour {hour}")

        # Get garage points within boundary
        filtered_garages = self.garages.gdf[self.garages.gdf.geometry.within(self.network.geometry.iloc[0])]

        if len(filtered_garages) == 0:
            logging.warning("No garages found within boundary")
            return None

        # Get the road network
        road_network = self.get_road_network()

        # Load and apply traffic factors
        traffic_data = TrafficData()
        traffic_factors = traffic_data.traffic_factors

        # Apply traffic factors to network
        logging.debug("Applying traffic factors to network")
        for u, v, k, data in road_network.edges(data=True, keys=True):
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
        for idx, garage in filtered_garages.iterrows():
            nearest_node = ox.nearest_nodes(
                road_network,
                garage.geometry.x,
                garage.geometry.y
            )
            garage_nodes[idx] = nearest_node

        # Calculate shortest paths using traffic-weighted times
        logging.debug("Calculating traffic-weighted shortest paths")
        assigner = nx.algorithms.voronoi.voronoi_cells(
            road_network,
            garage_nodes.values(),
            weight='traffic_time'
        )

        # Create region polygons from edge assignments
        voronoi_regions = {}
        for garage_id, cell_nodes in assigner.items():
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
                # Buffer each line slightly and union them
                buffer_distance = 0.0001  # Adjust based on coordinate system
                buffered_lines = [line.buffer(buffer_distance) for line in cell_edges]
                merged = unary_union(buffered_lines)

                # Simplify the merged polygon to remove artifacts
                polygon = merged.simplify(buffer_distance)

                if polygon.intersects(self.boundary_shape):
                    region = polygon.intersection(self.boundary_shape)
                    if not region.is_empty:
                        voronoi_regions[garage_id] = region

        logging.info(f"Created {len(voronoi_regions)} traffic-weighted Voronoi regions")
        return voronoi_regions

    def show_network(self, show_garages=False, show_roads=False, show_traffic=False, voronoi_type: VoronoiType = None, traffic_hour: int = None, coord: Tuple[float, float] = None):
        """
        Plot network visualisation with optional layers

        Args:
            show_garages (bool): Whether to show garage locations
            show_roads (bool): Whether to show the road network
            show_traffic (bool): Whether to show the traffic data
            voronoi_type (VoronoiType): Type of Voronoi diagram to display (if any)
        """
        logging.debug("Plotting network visualisation")

        ax = self.network.plot(edgecolor='black', facecolor='none')

        if voronoi_type:
            logging.debug(f"Adding {voronoi_type.value} Voronoi regions to plot")
            filtered_garages = self.garages.gdf[self.garages.gdf.geometry.within(self.network.geometry.iloc[0])]

            if voronoi_type == VoronoiType.NONE:
                pass
            elif voronoi_type == VoronoiType.EUCLIDEAN and self.euclidean_voronoi:
                for idx, region in enumerate(self.euclidean_voronoi.values()):
                    if isinstance(region, (shapely.Polygon, shapely.MultiPolygon)):
                        color = plt.cm.tab20(idx / len(filtered_garages))
                        gpd.GeoDataFrame(geometry=[region]).plot(
                            ax=ax,
                            color=color,
                            alpha=0.7,
                            edgecolor=color
                        )
            elif voronoi_type == VoronoiType.NETWORK and self.network_voronoi:
                for idx, region in enumerate(self.network_voronoi.values()):
                    if isinstance(region, (shapely.Polygon, shapely.MultiPolygon)):
                        color = plt.cm.tab20(idx / len(filtered_garages))
                        gpd.GeoDataFrame(geometry=[region]).plot(
                            ax=ax,
                            color=color,
                            alpha=1.0,
                            edgecolor=color
                        )
            elif voronoi_type == VoronoiType.TRAFFIC:
                if traffic_hour is None:
                    logging.warning("No hour specified for traffic Voronoi visualisation")
                    return

                logging.debug(f"Adding traffic-weighted Voronoi regions for hour {traffic_hour}")
                if traffic_hour in self.traffic_voronoi:
                    for idx, region in enumerate(self.traffic_voronoi[traffic_hour].values()):
                        if isinstance(region, (shapely.Polygon, shapely.MultiPolygon)):
                            color = plt.cm.tab20(idx / len(filtered_garages))
                            gpd.GeoDataFrame(geometry=[region]).plot(
                                ax=ax,
                                color=color,
                                alpha=0.7,
                                edgecolor=color
                            )
                else:
                    logging.warning(f"No traffic-weighted Voronoi regions available for hour {traffic_hour}")

        if show_roads:
            logging.debug("Adding road network to plot")
            road_network = self.get_road_network()
            roads_gdf = gpd.GeoDataFrame(geometry=[
                data['geometry'] for _, _, data in road_network.edges(data=True)
                if 'geometry' in data
            ])

            roads_gdf.plot(ax=ax, color='black', linewidth=0.5, alpha=0.3)

            if show_traffic and voronoi_type == VoronoiType.NONE:
                logging.debug("Adding traffic data visualisation")
                traffic_df = TrafficData().df
                
                # Filter points within boundary
                traffic_points = gpd.GeoDataFrame(
                    traffic_df,
                    geometry=gpd.points_from_xy(traffic_df.longitude, traffic_df.latitude),
                    crs="EPSG:4326"
                )
                mask = traffic_points.geometry.within(self.boundary_shape)
                filtered_traffic = traffic_points[mask].copy()
                
                if 'all_motor_vehicles' in filtered_traffic.columns:
                    # Create scatter plot with colormap for measurement points using crosses
                    scatter = ax.scatter(
                        filtered_traffic.longitude,
                        filtered_traffic.latitude,
                        c=filtered_traffic['all_motor_vehicles'],
                        cmap='RdYlGn_r',  # Red-Yellow-Green colormap (reversed)
                        marker='x',  # Cross marker
                        s=30,  # Smaller size
                        alpha=0.7,
                        zorder=2,
                        norm=plt.Normalize(
                            filtered_traffic['all_motor_vehicles'].quantile(0.1),
                            filtered_traffic['all_motor_vehicles'].quantile(0.9)
                        )
                    )
                    
                    # Color roads based on nearest traffic points
                    for _, road in roads_gdf.iterrows():
                        # Get road geometry points
                        if isinstance(road.geometry, LineString):
                            road_points = np.array(road.geometry.coords)
                        else:  # MultiLineString
                            road_points = np.array([p for line in road.geometry.geoms for p in line.coords])
                        
                        # Calculate distances to all traffic points for each road point
                        traffic_points_array = np.column_stack((filtered_traffic.longitude, filtered_traffic.latitude))
                        road_traffic_values = []
                        
                        for road_point in road_points:
                            distances = np.sqrt(np.sum((traffic_points_array - road_point) ** 2, axis=1))
                            # Use inverse distance weighting for the 3 nearest points
                            k = 3
                            nearest_indices = np.argpartition(distances, k)[:k]
                            weights = 1 / (distances[nearest_indices] ** 2)
                            weights = weights / np.sum(weights)  # normalize weights
                            road_traffic_values.append(
                                np.sum(weights * filtered_traffic['all_motor_vehicles'].iloc[nearest_indices])
                            )
                        
                        # Use average traffic value for the road segment
                        avg_traffic = np.mean(road_traffic_values)
                        
                        # Plot the road with color based on traffic using new colormap
                        road_color = plt.cm.RdYlGn_r(  # Red-Yellow-Green colormap (reversed)
                            plt.Normalize(
                                filtered_traffic['all_motor_vehicles'].quantile(0.1),
                                filtered_traffic['all_motor_vehicles'].quantile(0.9)
                            )(avg_traffic)
                        )
                        
                        if isinstance(road.geometry, LineString):
                            ax.plot(*road.geometry.xy, color=road_color, linewidth=2, alpha=0.7, zorder=1)
                        else:  # MultiLineString
                            for line in road.geometry.geoms:
                                ax.plot(*line.xy, color=road_color, linewidth=2, alpha=0.7, zorder=1)
                    
                    # Add colorbar
                    plt.colorbar(
                        scatter,
                        label='Traffic Volume (vehicles per day)',
                        ax=ax
                    )
                    
                    logging.debug(f"Added {len(filtered_traffic)} traffic measurement points and colored road network")
                else:
                    logging.warning("No traffic volume data available")

        if show_garages:
            logging.debug("Adding garage locations to plot")
            filtered_garages = self.garages.gdf[self.garages.gdf.geometry.within(self.network.geometry.iloc[0])]
            # Plot garages as small black dots
            filtered_garages.plot(
                ax=ax, 
                color='black',
                markersize=20,  # Smaller size
                marker='o',
                zorder=3
            )

        if coord:
            logging.debug(f"Adding coordinate {coord} to plot")
            # Plot as blue circle with proper size and zorder
            plt.scatter(
                coord[1],  # longitude (x)
                coord[0],  # latitude (y)
                color='blue',
                s=40,       # marker size
                zorder=4,
                label='Selected Location'
            )

        plt.gcf().set_size_inches(12, 8)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
        logging.debug("Network visualisation complete")
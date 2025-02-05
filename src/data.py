from pprint import pprint
import pandas as pd
import json
import os
import logging
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt

class TrafficData:
    def __init__(self, year: int = 2023):
        self.year = year
        logging.debug(f"Initialising TrafficData for year {year}")
        self.df = self.load_traffic_counts(year)
        self.traffic_factors = self.load_traffic_factors()

    def filter_traffic_counts(self, year: int = 2023):
        """
        Filter vehicle counts recorded on major and minor roads in the UK.

        Our model uses data for the 2023 year.

        Link to data: https://storage.googleapis.com/dft-statistics/road-traffic/downloads/data-gov-uk/dft_traffic_counts_raw_counts.zip
        """
        logging.debug(f"Starting traffic count filtering for {year}")
        assert 2000 <= year <= 2023, "Year must be between 2000 and 2023"

        df = pd.read_csv('data/dft_traffic_counts_raw_counts.csv', low_memory=False)
        logging.debug(f"Raw data loaded with {len(df)} rows")
        df = df[df['year'] == year]
        logging.info(f"Filtered to {len(df)} rows for year {year}")
        df.to_csv(f'data/dft_traffic_counts_{year}.csv', index=False)
        logging.debug(f"Saved filtered data to data/dft_traffic_counts_{year}.csv")

    def load_traffic_counts(self, year: int = 2023) -> pd.DataFrame:
        logging.debug(f"Attempting to load traffic counts for {year}")
        if not os.path.exists(f'data/dft_traffic_counts_{year}.csv'):
            logging.warning(f"Data file not found, generating new filtered dataset")
            self.filter_traffic_counts(year)

        df = pd.read_csv(f'data/dft_traffic_counts_{year}.csv', low_memory=False)
        logging.debug(f"Successfully loaded traffic data with {len(df)} rows")
        return df

    def map_road_type(self, road_name: str) -> str:
        if pd.isna(road_name):
            logging.debug("Mapping missing road name to 'unclassified'")
            return 'unclassified'
        if road_name.startswith('M'):
            return 'motorway'
        if road_name.startswith('A') and len(road_name) < 4:
            return 'trunk'
        if road_name.startswith('A'):
            return 'primary'
        if road_name.startswith('B'):
            return 'secondary'
        if road_name.startswith('C'):
            return 'tertiary'
        return 'unclassified'

    def load_traffic_factors(self) -> dict:
        logging.debug("Loading traffic factors")
        if os.path.exists('data/traffic_factors.json'):
            logging.info("Loading pre-calculated traffic factors from file")
            with open('data/traffic_factors.json', 'r') as f:
                return json.load(f)

        logging.warning("Generating new traffic factors from raw data")
        df = self.df
        df['road_type'] = df['road_name'].apply(self.map_road_type)
        logging.debug(f"Road types mapped: {df['road_type'].unique()}")

        hourly_avg = df.groupby(['road_type', 'hour'])['all_motor_vehicles'].mean().reset_index()
        logging.debug(f"Calculated hourly averages for {len(hourly_avg)} road-type/hour combinations")

        min_traffic = hourly_avg['all_motor_vehicles'].min()
        hourly_avg['weighting'] = hourly_avg['all_motor_vehicles'] / min_traffic
        logging.debug(f"Normalized weights (min traffic: {min_traffic})")

        TRAFFIC_FACTORS = {'default': 1.0}
        for _, row in hourly_avg.iterrows():
            road_type = row['road_type']
            hour = row['hour']
            weight = round(row['weighting'], 1)

            if road_type not in TRAFFIC_FACTORS:
                TRAFFIC_FACTORS[road_type] = {}

            TRAFFIC_FACTORS[road_type][hour] = weight

        logging.debug("Adding fallback values for missing hours/types")
        all_hours = range(7, 19)
        for rt in TRAFFIC_FACTORS:
            if rt == 'default':
                continue
            logging.debug(f"Processing fallbacks for {rt} road type")
            for hour in all_hours:
                if hour not in TRAFFIC_FACTORS[rt]:
                    # Use nearest available hour's value or default
                    TRAFFIC_FACTORS[rt][hour] = TRAFFIC_FACTORS.get(rt, {}).get(
                        max(h for h in TRAFFIC_FACTORS[rt].keys() if h <= hour), 1.0
                    )

        with open('data/traffic_factors.json', 'w') as f:
            json.dump(TRAFFIC_FACTORS, f)
        logging.info(f"Saved traffic factors to JSON file with {len(TRAFFIC_FACTORS)} road types")

        return TRAFFIC_FACTORS

    def plot_daily_traffic_counts(self):
        logging.debug("Creating traffic count visualisation")
        df = self.df

        df['road_type'] = df['road_name'].apply(self.map_road_type)
        logging.debug(f"Road types mapped: {df['road_type'].unique()}")

        hourly_avg = df.groupby(['road_type', 'hour', 'count_date'])['all_motor_vehicles'].mean().reset_index()
        logging.debug(f"Calculated hourly averages for {len(hourly_avg)} road-type/hour combinations")

        # Filter to dates with all 6 road types
        required_types = 6
        complete_dates = hourly_avg.groupby('count_date').filter(
            lambda g: g['road_type'].nunique() == required_types
        )

        if complete_dates.empty:
            raise ValueError("No dates with all 6 road types present")

        max_y = complete_dates['all_motor_vehicles'].max()
        fig = px.line(complete_dates, x='hour', y='all_motor_vehicles', color='road_type',
                      animation_frame='count_date',
                      title='Average Vehicle Counts by Road Type and Hour (Complete Days Only)',
                      labels={'hour': 'Hour of Day', 'all_motor_vehicles': 'Vehicle Count'},
                      category_orders={'road_type': ['motorway', 'trunk', 'primary',
                                                    'secondary', 'tertiary', 'unclassified']},
                      range_y=[-100, max_y],
                      markers=True)

        fig.update_xaxes(tickmode='array',
                        ticktext=[f'{h:02d}:00' for h in range(7,19)],
                        tickvals=list(range(7,19)))

        fig.show()

    def plot_average_traffic_counts(self):
        df = self.df
        df['road_type'] = df['road_name'].apply(self.map_road_type)
        logging.debug(f"Road types mapped: {df['road_type'].unique()}")

        # First average by date to get daily averages for each hour/road type
        daily_avg = df.groupby(['road_type', 'hour', 'count_date'])['all_motor_vehicles'].mean().reset_index()

        # Then average across dates for each hour/road type
        hourly_avg = daily_avg.groupby(['road_type', 'hour'])['all_motor_vehicles'].mean().reset_index()

        # Create the plot
        plt.rcParams.update({'font.size': 14})
        plt.rcParams.update({'font.family': 'Palatino'})
        plt.figure(figsize=(14, 6))  # Made figure wider to accommodate legend
        road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified']
        markers = ['o', 's', '^', 'D', 'v', 'p']

        for rt, marker in zip(road_types, markers):
            data = hourly_avg[hourly_avg['road_type'] == rt]
            plt.plot(data['hour'], data['all_motor_vehicles'],
                    marker=marker, label=rt, markersize=6)

        plt.title('Average Traffic Volume by Road Type and Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Vehicle Count')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.xticks(range(7,19), [f'{h:02d}:00' for h in range(7,19)])

        plt.tight_layout()
        plt.savefig('data/average_traffic_counts.png', dpi=300)
        plt.show()

class GarageData:
    def __init__(self):
        self.df = self.load_garage_data()
        self.gdf = self.load_garage_data_gdf()

    def load_garage_data(self) -> pd.DataFrame:
        logging.debug("Loading garage data")
        df = pd.read_csv('data/garage_coords.csv', low_memory=False)
        logging.debug(f"Successfully loaded garage data with {len(df)} rows")
        return df

    def load_garage_data_gdf(self) -> gpd.GeoDataFrame:
        logging.debug("Loading garage data as GeoDataFrame")
        df = self.df
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
        logging.debug(f"Successfully loaded garage data as GeoDataFrame with {len(gdf)} rows")
        return gdf

    def update_garage_location(self, postcode: str, lat: float, lon: float):
        """Update garage location in both DataFrame and GeoDataFrame"""
        # Update DataFrame
        self.df.loc[self.df['Postcode'] == postcode, 'Latitude'] = lat
        self.df.loc[self.df['Postcode'] == postcode, 'Longitude'] = lon

        # Update GeoDataFrame
        self.gdf = self.load_garage_data_gdf()  # Recreate GeoDataFrame with new coordinates
import pandas as pd
from pprint import pprint
import json

# Map road names to OSM road types based on UK road classification
def map_road_type(road_name):
    if pd.isna(road_name):
        return 'unclassified'
    if road_name.startswith('M'):
        return 'motorway'
    if road_name.startswith('A') and len(road_name) < 4:  # Major A-roads
        return 'trunk'
    if road_name.startswith('A'):
        return 'primary'
    if road_name.startswith('B'):
        return 'secondary'
    if road_name.startswith('C'):
        return 'tertiary'
    return 'unclassified'

# Load and preprocess data
df = pd.read_csv('data/dft_rawcount_local_authority_id_144_2023.csv', low_memory=False)
df['road_type'] = df['road_name'].apply(map_road_type)

# Calculate average traffic per road type per hour
hourly_avg = df.groupby(['road_type', 'hour'])['all_motor_vehicles'].mean().reset_index()

# Normalize traffic values to create weighting factors
min_traffic = hourly_avg['all_motor_vehicles'].min()
hourly_avg['weighting'] = hourly_avg['all_motor_vehicles'] / min_traffic

# Create traffic factor dictionary
TRAFFIC_FACTORS = {'default': 1.0}
for _, row in hourly_avg.iterrows():
    road_type = row['road_type']
    hour = row['hour']
    weight = round(row['weighting'], 1)

    if road_type not in TRAFFIC_FACTORS:
        TRAFFIC_FACTORS[road_type] = {}

    TRAFFIC_FACTORS[road_type][hour] = weight

# Add fallback values for missing hours/types
all_hours = range(7, 19)
for rt in TRAFFIC_FACTORS:
    if rt == 'default':
        continue
    for hour in all_hours:
        if hour not in TRAFFIC_FACTORS[rt]:
            # Use nearest available hour's value or default
            TRAFFIC_FACTORS[rt][hour] = TRAFFIC_FACTORS.get(rt, {}).get(
                max(h for h in TRAFFIC_FACTORS[rt].keys() if h <= hour), 1.0
            )

print("Generated Traffic Factors:")
pprint(TRAFFIC_FACTORS)


# # save to json
# with open('traffic_factors.json', 'w') as f:
#     json.dump(TRAFFIC_FACTORS, f)

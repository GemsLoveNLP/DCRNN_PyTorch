import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin
import argparse

# Function to calculate the distance between two lat/lon points using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6371.0
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    
    return R * c

def main(args):
    # Load the latitudes and longitudes from the CSV
    df = pd.read_csv(args.input_filename)  # Ensure it's in the format: sensor_id, latitude, longitude
    df = df[[args.sensor_column_name,
             args.lat_column_name,
             args.lon_column_name]]

    # Create a distance matrix
    n = len(df)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        lat1, lon1 = df.iloc[i][1], df.iloc[i][2]
        for j in range(i + 1, n):
            lat2, lon2 = df.iloc[j][1], df.iloc[j][2]
            dist = haversine(lat1, lon1, lat2, lon2)
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    sensor_ids = df[args.sensor_column_name].tolist()
    distances_list = []

    for i in range(n):
        for j in range(i + 1, n):
            from_sensor = sensor_ids[i]
            to_sensor = sensor_ids[j]
            distance = distance_matrix[i, j]
            distances_list.append([from_sensor, to_sensor, distance])

    # Create a DataFrame from the list
    distances_df = pd.DataFrame(distances_list, columns=['from', 'to', 'distance'])

    # Save to CSV
    distances_df.to_csv(args.output_filename, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_filename",
        type=str,
        default="data/addr_latlong.csv",
        help="Address CSV",
    )
    parser.add_argument(
        "--sensor_column_name",
        type=str,
        default="sensor_id",
        help="Name of sensor id column",
    )
    parser.add_argument(
        "--lat_column_name",
        type=str,
        default="latitude",
        help="Name of sensor id column",
    )
    parser.add_argument(
        "--lon_column_name",
        type=str,
        default="longitude",
        help="Name of sensor id column",
    )
    parser.add_argument(
        "--output_filename", 
        type=str, 
        default="data/distances.csv", 
        help="Output directory."
    )

    args = parser.parse_args()
    main(args)
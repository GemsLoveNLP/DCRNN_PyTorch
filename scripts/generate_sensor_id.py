import argparse
import numpy as np
import os
import pandas as pd

def main(args):
    df = pd.read_csv(args.input_filename)
    l = list(df[args.column_name])
    str = ",".join(l)
    with open(args.output_filename, 'w') as file:
        file.write(str)
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_filename",
        type=str,
        default="data/preprocess/addr_latlong.csv",
        help="Address CSV",
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default="sensor_id",
        help="Name of sensor id column",
    )
    parser.add_argument(
        "--output_filename", 
        type=str, 
        default="data/preprocess/sensor_id.txt", 
        help="Output directory."
    )

    args = parser.parse_args()
    main(args)
import pandas as pd
import os

# Load the dataframe from the original file
input_filename = "data/data.h5"
output_filename = "data/processed_data.h5"

# Load the DataFrame from the h5 file
df = pd.read_hdf(input_filename)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index("datetime")

# Store it
os.remove(output_filename)
store = pd.HDFStore(output_filename)
store.put('d',df)

store.close()

print(df)

import os
import glob

# Define the directory
directory = "models/"

# Get a list of all files in the directory
files = glob.glob(os.path.join(directory, "*"))

# Iterate and remove each file
for file in files:
    if os.path.isfile(file):  # Ensure it's a file
        os.remove(file)

print(f"All files in '{directory}' have been removed.")

import os
import glob
import shutil

def delete():
    # Define the directory
    directories = ("models/","experiments/","runs/experiments")

    for directory in directories:
        if os.path.exists(directory):  # Ensure the directory exists
            shutil.rmtree(directory)  # Remove directory and all contents
            os.makedirs(directory)  # Recreate the directory (optional)
            print(f"All contents in '{directory}' have been removed.")

def main():
    delete()

if __name__ == "__main__":
    main()

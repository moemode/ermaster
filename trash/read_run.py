from pathlib import Path
import json

# Define the path to the JSON file
file_path = Path("/home/v/coding/ermaster/runs/0_beer20_gpt-3.5-turbo-instruct.json")
# Check if the file exists
if file_path.is_file():
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
            # Now 'data' contains the contents of the JSON file as a dictionary
            print(data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")

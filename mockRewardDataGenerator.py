import json
import random

# Load the JSON data from the file
with open('synthetic_data.json', 'r') as f:
    data = json.load(f)

# Update the reward field with a random value between 0 and 3.5
for item in data:
    item['reward'] = round(random.uniform(0, 3.5), 2)  # round to 2 decimal places

# Save the updated data back to the JSON file
with open('synthetic_data.json', 'w') as f:
    json.dump(data, f, indent=4)
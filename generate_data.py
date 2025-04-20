
import json
import random

def generate_data(num_samples):
    titles = [f"outcome title {i + 1}" for i in range(num_samples)]
    descriptions = ["this is an outcome" for _ in range(num_samples)]
    data = []

    for i in range(num_samples):
        entry = {
            "id": i,
            "genre":  random.choice(['Drama', 'Action', 'Horror', 'Comedy']), #genre
            "companySuccessState": random.choice(['Engaged', 'Assigned', 'Inactive']),        #int cost  #country of origin
            "companySuccessInteractionScore": random.randint(0, 5),                           #popularity
            "companySuccessStatus": random.choice(['Red', 'Amber', 'Green']),                 #main actor
            "outcomes": random.randint(0, 9),  # Ensure category is within [0, 9]             #duration
            "reward": 0
        }
        data.append(entry)

    return data

# Generate and save synthetic data
num_samples = 1000
data = generate_data(num_samples)

# Save to JSON file
with open('synthetic_data.json', 'w') as f:
    json.dump(data, f, indent=4)

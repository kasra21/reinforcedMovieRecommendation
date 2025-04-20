
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
            "countryOrigin": random.choice(['US', 'UK', 'France']),        #int cost  #country of origin
            "cost": random.randint(1000, 9999999),                           #popularity duration
            "mainActor": random.choice(['Johnny Depp', 'Morgan Freeman', 'Tom Hanks']),                 #main actor
            "recommendatedForYou": random.randint(0, 9),  # Ensure category is within [0, 9], likeliness of liking the move    
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

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json


class Recommendation():
    def __init__(self, success_program: str, success_state: str, success_interaction_score: int, success_status: str, object_id: int):
        self.success_program = success_program
        self.success_state = success_state
        self.success_interaction_score = success_interaction_score
        self.success_status = success_state
        self.object_id = object_id
        self.predicted_outcome = get_predicted_outcome(success_program, success_state, success_interaction_score, success_status, object_id)

class Reward():
    def __init__(self, object_id: int, like: int):
        self.object_id = object_id
        self.reward = calculate_reward(object_id, like)

def check_and_add_id(data, new_entry=None, check_id=None):
    if check_id is not None:
        # Check if the id exists in the data
        for item in data:
            if item['id'] == check_id:
                print(f"ID {check_id} found: {item}")
                return item # ID found, no need to add
        print(f"ID {check_id} not found.")
    else:
        print("No ID provided.")

    # If no ID is given or ID not found, add the new entry to the data
    if new_entry is not None:
        last_id = max(item['id'] for item in data)
        print(last_id)
        new_entry['id'] = last_id+1
        data.append(new_entry)
        print(f"New entry added: {new_entry}")
    else:
        print("No new entry provided.")


# Predicting with New Data
def create_recommendation(rec_inputs):
    recommendation = Recommendation(rec_inputs.success_program, rec_inputs.success_state, rec_inputs.success_interaction_score, rec_inputs.success_status, rec_inputs.object_id)
    print(recommendation)
    return recommendation

# sets and gets the reward
def get_reward(reinforce_inputs):
    reward_object = Reward(reinforce_inputs.object_id, reinforce_inputs.like)
    return reward_object

#find the record by id, update the reward
def calculate_reward(object_id: int, like: int):
    with open('synthetic_data.json') as f:
        data = json.load(f)

    reward_to_return = 0
    for item in data:
            if item['id'] == object_id:
                print(f"ID {object_id} found: {item}")
                if item['reward'] == 0 and like == 0:
                    item['reward'] = 0
                    reward_to_return = 0
                else:
                    if like == 0:
                        item['reward'] = item['reward'] - 0.1
                    if like == 1:
                        item['reward'] = item['reward'] + 0.1
                    reward_to_return = item['reward']

                #save to database
                with open('synthetic_data.json', 'w') as f:
                    json.dump(data, f, indent=4)
                return reward_to_return

    print(f"ID {object_id} not found.")
    return None


def get_predicted_outcome(success_program: str, success_state: str, success_interaction_score: int, success_status: str, object_id: int):
    input_data = {
        "companySuccessProgram": success_program,
        "companySuccessState": success_state,
        "companySuccessInteractionScore": success_interaction_score,
        "companySuccessStatus": success_status,
        "reward": 0
    }

    # Define parameters for text processing
    max_success_program_length = 10
    max_success_state_length = 10
    max_success_status_length = 10
    num_words = 1000  # Example vocabulary size

    # Tokenizer for text fields
    success_program_tokenizer = Tokenizer(num_words=num_words)
    success_state_tokenizer = Tokenizer(num_words=num_words)
    success_status_tokenizer = Tokenizer(num_words=num_words)

    # Fit tokenizers on your training data (for demonstration, we use only the input data)
    success_program_tokenizer.fit_on_texts([input_data["companySuccessProgram"]])
    success_state_tokenizer.fit_on_texts([input_data["companySuccessState"]])
    success_status_tokenizer.fit_on_texts([input_data["companySuccessStatus"]])

    # Convert text to sequences
    success_program_seq = success_program_tokenizer.texts_to_sequences([input_data["companySuccessProgram"]])
    success_state_seq = success_state_tokenizer.texts_to_sequences([input_data["companySuccessState"]])
    success_status_seq = success_status_tokenizer.texts_to_sequences([input_data["companySuccessStatus"]])

    # Pad sequences
    success_program_pad = pad_sequences(success_program_seq, maxlen=max_success_program_length, padding='post')
    success_state_pad = pad_sequences(success_state_seq, maxlen=max_success_state_length, padding='post')
    success_status_pad = pad_sequences(success_status_seq, maxlen=max_success_status_length, padding='post')

    # Prepare the final input array
    model_input = np.concatenate([
        success_program_pad.astype(np.float32),
        success_state_pad.astype(np.float32),
        success_status_pad.astype(np.float32),
        np.array([[input_data["companySuccessInteractionScore"]]], dtype=np.float32)
    ], axis=1)

    # Load the model
    loaded_model = tf.saved_model.load('saved_model')
    infer = loaded_model.signatures['serving_default']

    # Make predictions
    predictions = infer(tf.convert_to_tensor(model_input))

    # Extract and print predictions
    # output = predictions['dense_1'].numpy()  # Adjust based on your model’s output tensor name
    output = predictions['output_0'].numpy()

    predicted_class = np.argmax(output, axis=-1)  # For classification tasks

    # Load data (much easier if we actually use a database, won't need to load this each time)
    with open('synthetic_data.json') as f:
        data = json.load(f)

    input_data['outcomes'] = int(str(predicted_class[0]))

    existing_item = check_and_add_id(data, new_entry=input_data, check_id=object_id)
    print(existing_item)
    # If you want to save the updated data back to a JSON file
    with open('synthetic_data.json', 'w') as f:
        json.dump(data, f, indent=4)

    if existing_item is not None:
        return str(existing_item['outcomes'])
    else:
        return str(predicted_class[0])



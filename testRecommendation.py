import json
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle


# Predicting with New Data

# Example input data
input_data = {
    "companySuccessProgram": "Guided",
    "companySuccessState": "Assigned",
    "companySuccessInteractionScore": 2,
    "companySuccessStatus": "Red"
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
loaded_model = tf.saved_model.load('retrained_model')
infer = loaded_model.signatures['serving_default']

# Make predictions
predictions = infer(tf.convert_to_tensor(model_input))

print(predictions.keys())   #these are the keys

# Extract and print predictions
# output = predictions['dense_1'].numpy()  # Adjust based on your model’s output tensor name
output = predictions['output_0'].numpy()

print(output)

predicted_class = np.argmax(output, axis=-1)  # For classification tasks

print("Predicted category:", predicted_class[0])
print(str(predicted_class[0]))


print([op.name for op in loaded_model.signatures['serving_default'].inputs])
print([op.name for op in loaded_model.signatures['serving_default'].outputs])

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
    "genre": "Drama",
    "countryOrigin": "US",
    "cost": 200000,
    "mainActor": "Johnny Depp"
}

# Define parameters for text processing
max_genre_length = 10
max_country_origin_length = 10
max_main_actor_length = 10
num_words = 1000  # Example vocabulary size

# Tokenizer for text fields
genre_tokenizer = Tokenizer(num_words=num_words)
country_origin_tokenizer = Tokenizer(num_words=num_words)
main_actor_tokenizer = Tokenizer(num_words=num_words)

# Fit tokenizers on your training data (for demonstration, we use only the input data)
genre_tokenizer.fit_on_texts([input_data["genre"]])
country_origin_tokenizer.fit_on_texts([input_data["countryOrigin"]])
main_actor_tokenizer.fit_on_texts([input_data["mainActor"]])

# Convert text to sequences
genre_seq = genre_tokenizer.texts_to_sequences([input_data["genre"]])
country_origin_seq = country_origin_tokenizer.texts_to_sequences([input_data["countryOrigin"]])
main_actor_seq = main_actor_tokenizer.texts_to_sequences([input_data["mainActor"]])

# Pad sequences
genre_pad = pad_sequences(genre_seq, maxlen=max_genre_length, padding='post')
country_origin_pad = pad_sequences(country_origin_seq, maxlen=max_country_origin_length, padding='post')
main_actor_pad = pad_sequences(main_actor_seq, maxlen=max_main_actor_length, padding='post')

# Prepare the final input array
model_input = np.concatenate([
    genre_pad.astype(np.float32),
    country_origin_pad.astype(np.float32),
    main_actor_pad.astype(np.float32),
    np.array([[input_data["cost"]]], dtype=np.float32)
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

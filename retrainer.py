import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import pickle

# Define custom loss function
def reward_adjusted_loss(reward):
    def loss(y_true, y_pred):
        base_loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
        
        # Ensure reward_weight is the same dtype as base_loss
        reward_weight = tf.reduce_mean(tf.cast(reward, tf.float32))
        
        return base_loss * reward_weight
    return loss

# Load the model
model = load_model('model.h5')

# Prepare your data
with open('synthetic_data.json') as f:
        data = json.load(f)
# data = {
#     "genre": ['Drama', 'Action', 'Horror', 'Comedy'],
#     "countryOrigin": ['US', 'UK', 'France'],
#     "cost": [5000, 2000, 3000, 4000, 1000],
#     "mainActor": ['Johnny Depp', 'Morgan Freeman', 'Tom Hanks', 'Johnny Depp', 'Morgan Freeman'],
#     "recommendatedForYou": [0, 2, 2, 1, 3],
#     "reward": [4.5, 1.2, 3.0, 4.8, 0.5]
# }

df = pd.DataFrame(data)

# Prepare text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['genre'].tolist() + df['countryOrigin'].tolist() + df['mainActor'].tolist())
genre_sequences = tokenizer.texts_to_sequences(df['genre'])
country_origin_sequences = tokenizer.texts_to_sequences(df['countryOrigin'])
main_actor_sequences = tokenizer.texts_to_sequences(df['mainActor'])

# Pad sequences
max_len_genre = 10
max_len_country_origin = 10
max_len_main_actor = 10

genre_padded = pad_sequences(genre_sequences, maxlen=max_len_genre)
country_origin_padded = pad_sequences(country_origin_sequences, maxlen=max_len_country_origin)
main_actor_padded = pad_sequences(main_actor_sequences, maxlen=max_len_main_actor)

# Prepare other features

# Save to JSON file
# with open('rewarded_synthetic_data.json', 'w') as f:
#     json.dump(data, f, indent=4)

# # Load data
# with open('rewarded_synthetic_data.json') as f:
#     data = json.load(f)

    
# Tokenize and pad text data
tokenizer_genre = Tokenizer()
tokenizer_genre.fit_on_texts(df['genre'])
genre_sequences = tokenizer_genre.texts_to_sequences(df['genre'])
genre_padded = pad_sequences(genre_sequences, maxlen=10)

tokenizer_country_origin = Tokenizer()
tokenizer_country_origin.fit_on_texts(df['countryOrigin'])
country_origin_sequences = tokenizer_country_origin.texts_to_sequences(df['countryOrigin'])
country_origin_padded = pad_sequences(country_origin_sequences, maxlen=10)

tokenizer_main_actor = Tokenizer()
tokenizer_main_actor.fit_on_texts(df['mainActor'])
main_actor_sequences = tokenizer_main_actor.texts_to_sequences(df['mainActor'])
main_actor_padded = pad_sequences(main_actor_sequences, maxlen=10)

X = np.hstack((genre_padded, country_origin_padded, main_actor_padded, df[['cost']].values))
recommendatedForYou = np.array(df['recommendatedForYou'])
reward = np.array(df['reward'])

# Save Tokenizers
with open('tokenizer_title.pkl', 'wb') as f:
    pickle.dump(tokenizer_genre, f)
with open('tokenizer_country_origin.pkl', 'wb') as f:
    pickle.dump(tokenizer_country_origin, f)
with open('tokenizer_main_actor.pkl', 'wb') as f:
    pickle.dump(tokenizer_main_actor, f)

# Compile and fit the model
model.compile(optimizer='adam', loss=reward_adjusted_loss(reward), metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(
    X, 
    recommendatedForYou,
    epochs=10,
    batch_size=2,
    validation_split=0.2
)

# Save the retrained model
model.save('retrained_model.h5')

# Convert the model to a .pb file (TensorFlow SavedModel format)
tf.saved_model.save(model, 'retrained_model')

print("Model retrained and saved as 'retrained_model'.")
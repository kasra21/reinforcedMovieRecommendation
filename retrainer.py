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
#     "companySuccessProgram": ['Core', 'Guided', 'Advanced', 'Total', 'Guided'],
#     "companySuccessState": ['Engaged', 'Assigned', 'Inactive', 'Engaged', 'Assigned'],
#     "companySuccessInteractionScore": [5, 2, 3, 4, 1],
#     "companySuccessStatus": ['Red', 'Amber', 'Green', 'Amber', 'Green'],
#     "outcomes": [0, 2, 2, 1, 3],
#     "reward": [4.5, 1.2, 3.0, 4.8, 0.5]
# }

df = pd.DataFrame(data)

# Prepare text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['companySuccessProgram'].tolist() + df['companySuccessState'].tolist() + df['companySuccessStatus'].tolist())
success_program_sequences = tokenizer.texts_to_sequences(df['companySuccessProgram'])
success_state_sequences = tokenizer.texts_to_sequences(df['companySuccessState'])
success_status_sequences = tokenizer.texts_to_sequences(df['companySuccessStatus'])

# Pad sequences
max_len_success_program = 10
max_len_success_state = 10
max_len_success_status = 10

success_program_padded = pad_sequences(success_program_sequences, maxlen=max_len_success_program)
success_state_padded = pad_sequences(success_state_sequences, maxlen=max_len_success_state)
success_status_padded = pad_sequences(success_status_sequences, maxlen=max_len_success_status)

# Prepare other features

# Save to JSON file
# with open('rewarded_synthetic_data.json', 'w') as f:
#     json.dump(data, f, indent=4)

# # Load data
# with open('rewarded_synthetic_data.json') as f:
#     data = json.load(f)

    
# Tokenize and pad text data
tokenizer_success_program = Tokenizer()
tokenizer_success_program.fit_on_texts(df['companySuccessProgram'])
success_program_sequences = tokenizer_success_program.texts_to_sequences(df['companySuccessProgram'])
success_program_padded = pad_sequences(success_program_sequences, maxlen=10)

tokenizer_success_state = Tokenizer()
tokenizer_success_state.fit_on_texts(df['companySuccessState'])
success_state_sequences = tokenizer_success_state.texts_to_sequences(df['companySuccessState'])
success_state_padded = pad_sequences(success_state_sequences, maxlen=10)

tokenizer_success_status = Tokenizer()
tokenizer_success_status.fit_on_texts(df['companySuccessStatus'])
success_status_sequences = tokenizer_success_status.texts_to_sequences(df['companySuccessStatus'])
success_status_padded = pad_sequences(success_status_sequences, maxlen=10)

X = np.hstack((success_program_padded, success_state_padded, success_status_padded, df[['companySuccessInteractionScore']].values))
outcomes = np.array(df['outcomes'])
reward = np.array(df['reward'])

# Save Tokenizers
with open('tokenizer_title.pkl', 'wb') as f:
    pickle.dump(tokenizer_success_program, f)
with open('tokenizer_success_state.pkl', 'wb') as f:
    pickle.dump(tokenizer_success_state, f)
with open('tokenizer_success_status.pkl', 'wb') as f:
    pickle.dump(tokenizer_success_status, f)

# Compile and fit the model
model.compile(optimizer='adam', loss=reward_adjusted_loss(reward), metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(
    X, 
    outcomes,
    epochs=10,
    batch_size=2,
    validation_split=0.2
)

# Save the retrained model
model.save('retrained_model.h5')

# Convert the model to a .pb file (TensorFlow SavedModel format)
tf.saved_model.save(model, 'retrained_model')

print("Model retrained and saved as 'retrained_model'.")
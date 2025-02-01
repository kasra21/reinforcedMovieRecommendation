# Generate Synthetic Data

import json
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


# Load data
with open('synthetic_data.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)


# Preprocess the data
def preprocess_data(df):
    # Encode categorical variables

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
    success_status_sequences = tokenizer_success_state.texts_to_sequences(df['companySuccessStatus'])
    success_status_padded = pad_sequences(success_status_sequences, maxlen=10)

    X = np.hstack((success_program_padded, success_state_padded, success_status_padded, df[['companySuccessInteractionScore']].values))
    y = df['outcomes'].values

    return train_test_split(X, y, test_size=0.2, random_state=42), tokenizer_success_program, tokenizer_success_state, tokenizer_success_status


(X_train, X_test, y_train, y_test), tokenizer_success_program, tokenizer_success_state, tokenizer_success_status = preprocess_data(df)

# Feature scaling
scaler = StandardScaler()
X_train[:, -2:] = scaler.fit_transform(X_train[:, -2:])
X_test[:, -2:] = scaler.transform(X_test[:, -2:])

# Save Tokenizers
with open('tokenizer_success_program.pkl', 'wb') as f:
    pickle.dump(tokenizer_success_program, f)
with open('tokenizer_success_state.pkl', 'wb') as f:
    pickle.dump(tokenizer_success_state, f)
with open('tokenizer_success_status.pkl', 'wb') as f:
    pickle.dump(tokenizer_success_status, f)


# Build and Train the Model
def create_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')  # Output layer for 10 categories
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Assuming input_shape is the number of features after preprocessing
input_shape = X_train.shape[1]
model = create_model(input_shape)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')

print(model.summary)

# Save the model
model.save('model.h5')

# Convert the model to a .pb file (TensorFlow SavedModel format)
tf.saved_model.save(model, 'saved_model')


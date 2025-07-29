# save_dummy_lstm_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Dummy LSTM model
model = Sequential([
    LSTM(32, input_shape=(10, 1)),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save model
model.save("lstm_movement_model.keras")
print("Model saved as lstm_movement_model.keras")

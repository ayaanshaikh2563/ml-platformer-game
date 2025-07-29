import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Load data
# Make sure movement_data.csv has at least 'x_position'
data = pd.read_csv("movement_data.csv")

# Extract features and labels - now ONLY x_position for simplicity and matching game
# If 'jump' column is in your CSV, you can ignore it for this model,
# or you can filter the data to only include x_position if that's all you want to model.
positions = data[["x_position"]].values # Changed to only use x_position

# Normalize
# We need a scaler that works for a single feature (x_position)
scaler = MinMaxScaler(feature_range=(0, 1)) # Explicitly setting feature_range
positions_scaled = scaler.fit_transform(positions)

# Create sequences
def create_sequences(data, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length]) # Predict the next single x_position
    return np.array(X), np.array(y)

X, y = create_sequences(positions_scaled)

print(f"Shape of X (sequences): {X.shape}") # Should be (num_samples, sequence_length, 1)
print(f"Shape of y (targets): {y.shape}")   # Should be (num_samples, 1)

# Model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])), # input_shape will be (10, 1)
    Dense(1) # Predict only x_position
])

model.compile(optimizer="adam", loss="mse")

# Train
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, # Increased epochs
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]) # Increased patience

# Save
model.save("enemy_lstm_model.keras")
# Save min/max of the scaler.data_min_ and scaler.data_max_ for the single feature
np.save("scaler_min.npy", scaler.data_min_)
np.save("scaler_max.npy", scaler.data_max_)

print("✅ Model trained and saved as enemy_lstm_model.keras")
print("✅ Scaler min/max values saved as scaler_min.npy and scaler_max.npy")
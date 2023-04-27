import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load the dataset
dataset = pd.read_csv('air_quality.csv')

# Split the data into training and testing sets
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train_data = dataset.iloc[:train_size, :]
test_data = dataset.iloc[train_size:, :]

# Scale the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Prepare the data for training
def prepare_data(data, timesteps):
    X, y = [], []
    for i in range(len(data)-timesteps-1):
        X.append(data[i:(i+timesteps), :])
        y.append(data[(i+timesteps), 0])
    return np.array(X), np.array(y)

timesteps = 24
X_train, y_train = prepare_data(train_data, timesteps)
X_test, y_test = prepare_data(test_data, timesteps)

# Create the Nested LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(timesteps, X_train.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

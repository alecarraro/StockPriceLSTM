#model_training.py
#With this file you can train the model with the selected parameters

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import joblib
import yaml
from data_preparation import prepare_data, load_config

# Assemble the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=125, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(units=125, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(units=125))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Train LSTM model
def train_model(X_train, y_train, X_validate, y_validate, model_path, epochs=50, batch_size=64):
    model = build_model((X_train.shape[1], X_train.shape[2]))
    checkpoint = ModelCheckpoint(
        model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=0)

    history = model.fit(
        X_train, y_train, validation_data=(X_validate, y_validate),
        epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    
    return model, history

# Plot training & validation loss
def plot_model_performance(history):
    plt.figure(figsize=(18, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM Model Performance")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Prepare data, build model and train
def main():
    prepare_data("config.yaml")
    config = load_config("config.yaml")

    X_train, y_train, X_validate, y_validate, _, _ = prepare_data("config.yaml")

    # Training params
    model_training_config = config['model_training']
    model_path = model_training_config['model_path']
    epochs = model_training_config['epochs']
    batch_size = model_training_config['batch_size']

    # Train model
    model, history = train_model(X_train, y_train, X_validate, y_validate, model_path, epochs, batch_size)
    plot_model_performance(history)

# If this file is run, call the main training function
if __name__ == "__main__":
    main()

# model_prediction.py
# with this file you can run the model to predict on the test set

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib
from data_preparation import prepare_data, load_config
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime, timedelta

# Load model and scaler
def load_model_and_scaler(model_path, scaler_path):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Inverse and transform prediction
def inverse_transform(scaler, data, sequence_size=60):
    # Dummy columns (same number as features) to match original data shape for inverse transform
    dummy_data = np.ones((len(data), scaler.n_features_in_))
    dummy_data[:, 0] = data.squeeze()  # Replace the first column with actual predicted data
    return scaler.inverse_transform(dummy_data)[:, 0]


# Plot the predicted vs real prices
def plot_predictions(real_prices, predicted_prices, dates, title="Stock Price Prediction"):
    plt.figure(figsize=(12, 6))
    
    plt.plot(dates, real_prices, color='blue', label='Real Stock Price')
    plt.plot(dates, predicted_prices, color='red', label='Predicted Stock Price')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45) 
    plt.tight_layout() 
    plt.show()

# Predict steps
def main():
    config = load_config('config.yaml')
    _, _, _, _, X_test, y_test = prepare_data()

    model_path = config['model_training']["model_path"]
    scaler_path = config["scaler_save_path"]
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    y_test_pred = model.predict(X_test)

    y_test_inv = inverse_transform(scaler, y_test)
    y_test_pred_inv = inverse_transform(scaler, y_test_pred)

    test_start_date = datetime.strptime(config['dates']['test_end'], '%Y-%m-%d') - timedelta(days=len(y_test_inv))
    dates = [test_start_date + timedelta(days=i) for i in range(len(y_test_inv))]

    plot_predictions(y_test_inv, y_test_pred_inv, dates, title="Stock Price Prediction vs Actual")

if __name__ == "__main__":
    main()

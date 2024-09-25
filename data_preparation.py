# data_preparation.py
# all the functions you need to preprocess the data
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import yaml

# Yahoo Finance API override
yf.pdr_override()

# Function to load and split data
def load_and_split_data(ticker, start, end, train_start, train_end, val_start, val_end, test_start, test_end):
    dataset = pdr.get_data_yahoo(ticker, start=start, end=end)
    
    # Split into training, validation, and test sets
    train_data = dataset.loc[train_start:train_end]
    validation_data = dataset.loc[val_start:val_end]
    test_data = dataset.loc[test_start:test_end]
    
    return train_data, validation_data, test_data, dataset

# Data scaler
def scale_data(train_data, validation_data, test_data, save_path=None):
    sc = MinMaxScaler(feature_range=(0, 1))
    
    # Scale the datasets
    train_scaled = sc.fit_transform(train_data)
    validation_scaled = sc.transform(validation_data)
    test_scaled = sc.transform(test_data)

    if save_path:
        joblib.dump(sc, save_path)
    
    return train_scaled, validation_scaled, test_scaled, sc

# Data splitter
def plot_data_split(train_data, validation_data, test_data):
    plt.figure(figsize=(18, 6))
    plt.plot(train_data["Open"], color="cornflowerblue")
    plt.plot(validation_data["Open"], color="orange")
    plt.plot(test_data["Open"], color="green")
    plt.legend(["Train Data", "Validation Data", "Test Data"])
    plt.title("Data Split for Google Stock Price")
    plt.xlabel("Samples Over Time")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

# Prepare data for LSTM
def construct_lstm_data(data, sequence_size, target_attr_idx):
    data_X, data_y = [], []
    for i in range(sequence_size, len(data)):
        data_X.append(data[i-sequence_size:i, 0:data.shape[1]])
        data_y.append(data[i, target_attr_idx])
    return np.array(data_X), np.array(data_y)

# Load the configuration from the YAML file
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Pipeline for data preparation
def prepare_data(yaml_file='config.yaml'):

    config = load_config(yaml_file)
    ticker = config['ticker']
    dates = config['dates']
    sequence_size = config['sequence_size']
    scaler_save_path = config['scaler_save_path']

    start_date = dates['start']
    end_date = dates['end']
    
    train_start = dates['train_start']
    train_end = dates['train_end']
    val_start = dates['val_start']
    val_end = dates['val_end']
    test_start = dates['test_start']
    test_end = dates['test_end']

    train_data, validation_data, test_data, dataset = load_and_split_data(
        ticker, start_date, end_date, train_start, train_end, val_start, val_end, test_start, test_end)

    train_scaled, validation_scaled, test_scaled, scaler = scale_data(
        train_data, validation_data, test_data, scaler_save_path)

    # Optional: plot split
    #plot_data_split(train_data, validation_data, test_data)

    X_train, y_train = construct_lstm_data(train_scaled, sequence_size, 0)
    data_all_scaled = np.concatenate([train_scaled, validation_scaled, test_scaled], axis=0)
    
    train_size = len(train_scaled)
    validate_size = len(validation_scaled)
    test_size = len(test_scaled)

    X_validate, y_validate = construct_lstm_data(data_all_scaled[train_size-sequence_size:train_size+validate_size, :], sequence_size, 0)
    X_test, y_test = construct_lstm_data(data_all_scaled[-(test_size+sequence_size):, :], sequence_size, 0)

    return X_train, y_train, X_validate, y_validate, X_test, y_test

if __name__ == "__main__":
    prepare_data("config.yaml")

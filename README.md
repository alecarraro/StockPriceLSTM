# LSTM Stock Price Prediction

This project utilizes Long Short-Term Memory (LSTM) neural networks to predict stock prices based on historical data and backtest trading strategies.
## Project Structure

The project consists of three main scripts:

1. **data_preparation.py**
   - This script is responsible for loading, preprocessing, and splitting the stock price data into training, validation, and test sets. It utilizes the Yahoo Finance API to fetch historical stock prices and scales the data using MinMaxScaler for better performance during training.

2. **model_training.py**
   - This script defines the architecture of the LSTM model and trains it using the prepared datasets. It includes model evaluation to assess performance using the validation set.

3. **model_prediction.py**
   - This script loads the trained model and scaler, predicts future stock prices, and visualizes the predictions compared to the actual stock prices.

The project uses a YAML configuration file (`config.yaml`) to specify the parameters to be used, including ticker, dates and training parameters.

## Dataset

The dataset used in this project is sourced from Yahoo Finance. In this particular case, the stock price data (Open) for NVIDIA (ticker: NVDA) was used.

## Results

To test the implementation I used three LSTM layers, each with 125 units and a dropout rate of 0.2, trained on 80 epochs on a historical data from 2019 to 2024. The result on the testing set is displayed below:
![results NVDA](https://github.com/user-attachments/assets/1212200a-1161-43c6-bad0-7dd6731cbe06)



## Requirements

To run this project, ensure you must have the libraries listed in the `reqs.txt`file

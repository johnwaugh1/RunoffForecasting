# Runoff Forecasting Project for CS-4440

This project aims to improve the accuracy of National Water Model (NWM) runoff forecasts by using a Long Short-Term Memory (LSTM) neural network to post-process and correct NWM forecast errors. The model is trained to predict the residuals (errors) between NWM forecasts and observed streamflow from USGS, and applies these predictions to improve the original forecast.

## Requirements

Install dependencies with:

py -m pip install <library>

Major libraries used:
    numpy
    pandas
    matplotlib
    scikit-learn
    tensorflow
    pyarrow (for reading Parquet files)

## Usage

1. Clone the repository
2. Prepare the data
3. Train the model
4. Evaluate the model

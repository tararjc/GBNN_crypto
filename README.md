# GBNN_crypto

Gradient Boosted Neural Network to predict Cryptocurrency price

<hr>

This repository contains Python code for predicting stock prices using three different models: SARIMA (Seasonal Autoregressive Integrated Moving Average), GBNN (Gradient Boosted Neural Network), and MLP (Multi-Layer Perceptron). The code also includes feature selection using Support Vector Regression (SVR).

# Author
Taraneh Shahin 
orcid: 0000-0001-8381-8462

## Overview

The repository includes following main scripts:

1. **`train_close.py` and `train_vol.py`** : These scripts performs the following tasks:

* Reads stock price data from CSV files in the specified directory.
* Computes various technical indicators using the `BaseIndicators` class.
* Performs feature selection using SVR through the `FeatuerSelection` class.
* Splits the data into training and testing sets.
* Trains SARIMA, GBNN, and MLP models on the training set.
* Evaluates the models on the testing set and calculates metrics (RMSE, MAE, MSE, R2).
* Plots true vs predicted values for each model.
* Stores the results and plots in the "results_close_price" directory.

2. **`base_indicators.py` and `featuer_selection.py`** : These scripts contain the `BaseIndicators` and `FeatuerSelection` classes, which are utilized in the main script for computing technical indicators and performing feature selection, respectively.

## Requirements

Make sure you have the required libraries installed. You can install them using the following:

```
https://github.com/GAA-UAM/GBNN
```

## Usage

1. Place your stock price CSV files in the "Dataset" directory.
2. Run the train_close `.py ` script:

Dependencies

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `statsmodels`
* `GBNN` (Make sure the clone "GBNN" module is available in the specified path.)

# Release Date

01.Aug.2023

# Update Date

10.Dec.2023

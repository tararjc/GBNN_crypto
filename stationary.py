# %%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf


def check_stationarity_autocorrelation(time_series, max_lag=20):
    result = adfuller(time_series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")

    # Plot autocorrelation function (ACF)
    plot_acf(time_series, lags=max_lag)
    plt.title("Autocorrelation Function (ACF)")
    plt.show()

    if result[1] > 0.05:
        print("\nApplying differencing to achieve stationarity...")
        diff_order = 1
        differenced_series = time_series.diff(diff_order).dropna()

        # Check stationarity of differenced series
        result_diff = adfuller(differenced_series)
        print(f"\nADF Statistic (after differencing): {result_diff[0]}")
        print(f"p-value: {result_diff[1]}")
        print("Critical Values:")
        for key, value in result_diff[4].items():
            print(f"   {key}: {value}")

        # Plot ACF of differenced series
        plot_acf(differenced_series, lags=max_lag)
        plt.title("Autocorrelation Function (ACF) - After Differencing")
        plt.show()

        return differenced_series
    else:
        print("\nThe time series is stationary.")
        return time_series


def check_stationarity_autocorrelation(time_series, max_lag=20):
    result = adfuller(time_series)

    results_df = pd.DataFrame(
        {
            "Dataset": [file[8:-4]],
            "ADF Statistic": [result[0]],
            "p-value": [result[1]],
            "Critical Value (1%)": [result[4]["1%"]],
            "Critical Value (5%)": [result[4]["5%"]],
            "Critical Value (10%)": [result[4]["10%"]],
        }
    )

    return results_df


def make_stationary_and_save(file):
    new_df = pd.read_csv(file)
    new_df.sort_values(by=["Date"], inplace=True)
    time_series = new_df.Close

    differenced_series = time_series.diff().dropna()

    output_file = os.path.join("Stationary_Dataset", f"{file[8:]}")
    new_df["Close"] = differenced_series
    new_df.to_csv(output_file, index=False)

    results_df = check_stationarity_autocorrelation(differenced_series)

    return results_df


directory = "Dataset"
rmse_dict = {}
csv_files = glob.glob(os.path.join(directory, "*.csv"))

output_file = r"results_close_price\results.csv"
all_results_df_before_converting = pd.DataFrame()
all_results_df_after_converting = pd.DataFrame()

for file in csv_files:
    new_df = pd.read_csv(file)
    new_df.sort_values(by=["Date"], inplace=True)
    time_series = new_df.Close
    results_df = check_stationarity_autocorrelation(time_series)
    all_results_df_before_converting = pd.concat(
        [all_results_df_before_converting, results_df], ignore_index=True
    )
    stationary_file = make_stationary_and_save(file)
    all_results_df_after_converting = pd.concat(
        [all_results_df_after_converting, stationary_file], ignore_index=True
    )
    print(f"Stationary time series saved to: {stationary_file}")

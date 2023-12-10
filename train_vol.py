import sys
import warnings
warnings.simplefilter("ignore")
sys.path.append("GBNN")
import os
import glob
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from Base.base_indicators import BaseIndicators
from Base.featuer_selection import FeatuerSelection_vol
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from statsmodels.tsa.statespace.sarimax import SARIMAX
from GBNN import GNEGNERegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
)
from Base.utils import *
from sklearn.neural_network import MLPRegressor


directory = "Dataset"
rmse_dict = {}
csv_files = glob.glob(os.path.join(directory, "*.csv"))

output_file = r"results_volatility\results.csv"

with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["File", "Method", "RMSE", "MAE", "MSE", "R2"])

    for file in tqdm(csv_files):
        new_df = pd.read_csv(file)

        model = BaseIndicators(new_df)
        methods = [
            (model._rsi, "rsi"),
            (model._stoch, "stoch"),
            (model._cci, "cci"),
            (model._adx, "adx"),
            (model._macd, "macd"),
            (model._traded_value, "value"),
            (model._Volatility, "volatility"),
            (model._Momentum, "Momentum"),
            (model._ma, "MA"),
            (model._ema, "EMA"),
            (model._roc, "ROC"),
        ]

        i = 0
        while i < len(methods):
            for func, func_name in methods:
                new_df[str(func_name)] = func().values
                i += 1
        new_df.fillna(0, inplace=True)
        new_df["volatility"] = new_df["Close"].pct_change().rolling(5).std() * (
            365**0.5
        )
        new_df.dropna(inplace=True)
        selector = FeatuerSelection_vol(new_df)
        df = selector()
        X = df.drop(["volatility"], axis=1)
        y = df["volatility"]

        X = X.reset_index()
        X.drop(columns="Date", inplace=True)
        try:
            X.drop(columns=["Open", "High", "Low", "Adj Close"], inplace=True)
        except:
            pass
        scaler = StandardScaler()
        y = scaler.fit_transform(y.values.reshape(-1, 1))
        y = pd.Series(np.squeeze(y))
        y = np.array(y)
        y = pd.Series(y, index=df.index)

        X = scaler.fit_transform(X)

        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        sarima_model = SARIMAX(y_train, order=(1, 0, 0), seasonal_order=(0, 1, 1, 12))
        model_fit = sarima_model.fit(disp=False)

        y_pred = model_fit.predict(
            start=len(X_train), end=len(X_train) + len(X_test) - 1
        )
        pred_sarima = y_pred

        gbnn_model = GNEGNERegressor(max_iter=200, activation="relu")
        param_grid_gbnn = {
            "num_nn_step": [1, 2, 3],
            "eta": [0.025, 0.05, 0.75, 1],
            "subsample": [0.5, 0.75, 1],
        }
        grid_search_gbnn = GridSearchCV(
            gbnn_model, param_grid_gbnn, scoring="r2", cv=kf, n_jobs=-1, refit=True
        )
        grid_search_gbnn.fit(X_train, y_train)
        best_gbnn_model = grid_search_gbnn.best_estimator_
        pred_gbnn = best_gbnn_model.predict(X_test)

        MLP_model = MLPRegressor(early_stopping=True, activation="relu", max_iter=200)
        param_grid_mlp = {"learning_rate": ["constant", "invscaling"]}
        grid_search_mlp = GridSearchCV(
            MLP_model, param_grid_mlp, scoring="r2", cv=kf, n_jobs=-1, refit=True
        )
        grid_search_mlp.fit(X_train, y_train)
        best_mlp_model = grid_search_mlp.best_estimator_
        y_pred_MLP = best_mlp_model.predict(X_test)

        MLP_model = MLPRegressor(early_stopping=True, activation="logistic")

        rmse_s = mean_squared_error(y_test.values, y_pred.values, squared=False)
        MAE_s = mean_absolute_error(y_test.values, y_pred.values)
        MSE_s = mean_squared_error(y_test.values, y_pred.values)
        r2_s = explained_variance_score(y_test.values, y_pred.values)

        print("-------------------------------------")
        print(f"File: {extract_string(file, pattern)}")
        writer.writerow(
            [extract_string(file, pattern), "SARIMA", rmse_s, MAE_s, MSE_s, r2_s]
        )

        rmse_g = mean_squared_error(y_test.values, pred_gbnn, squared=False)
        MAE_g = mean_absolute_error(y_test.values, pred_gbnn)
        MSE_g = mean_squared_error(y_test.values, pred_gbnn)
        r2_g = explained_variance_score(y_test.values, pred_gbnn)
        writer.writerow(
            [extract_string(file, pattern), "GBNN", rmse_g, MAE_g, MSE_g, r2_g]
        )

        rmse_mlp = mean_squared_error(y_test.values, y_pred_MLP, squared=False)
        MAE_mlp = mean_absolute_error(y_test.values, y_pred_MLP)
        MSE_mlp = mean_squared_error(y_test.values, y_pred_MLP)
        r2_mlp = explained_variance_score(y_test.values, y_pred_MLP)
        writer.writerow(
            [extract_string(file, pattern), "MLP", rmse_mlp, MAE_mlp, MSE_mlp, r2_mlp]
        )

        data_gbnn = {"True": y_test.values, "Predicted": pred_gbnn}
        df_gbnn = pd.DataFrame(data_gbnn, index=y_test.index)

        # Create the second DataFrame
        data_sarima = {"True": y_test.values, "Predicted": pred_sarima}
        df_sarima = pd.DataFrame(data_sarima, index=y_test.index)

        data_mlp = {"True": y_test, "Predicted": y_pred_MLP}
        df_mlp = pd.DataFrame(data_mlp, index=y_test.index)

        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=1, ncols=3, sharex=False, figsize=(20, 5)
        )

        df_gbnn.plot(ax=ax1)
        ax1.set_title("GBNN model")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Volatility trend")

        df_sarima.plot(ax=ax2)
        ax2.set_title("SARIMA model")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volatility trend")

        df_mlp.plot(ax=ax3)
        ax3.set_title("MLP model")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Volatility trend")

        fig.suptitle("True vs Predicted values")
        plt.tight_layout()

        fig.savefig(
            rf"results_volatility\{extract_string(file, pattern)}_Plot.png", dpi=700
        )

        fig, ax = plt.subplots(1, 3, figsize=(20, 5))

        residuals = y_test - pred_gbnn
        residuals = [r - p for r, p in zip(y_test, pred_gbnn)]

        ax[0].scatter(y_test, residuals)
        ax[0].axhline(
            0, color="red", linestyle="--"
        )  # Add a horizontal line at y=0 for reference
        ax[0].set_xlabel("Real Values")
        ax[0].set_ylabel("Residuals")
        ax[0].set_title("GBNN")

        residuals = y_test - pred_gbnn
        residuals = [r - p for r, p in zip(y_test, pred_sarima)]

        ax[1].scatter(y_test, residuals)
        ax[1].axhline(
            0, color="red", linestyle="--"
        )  # Add a horizontal line at y=0 for reference
        ax[1].set_xlabel("Real Values")
        ax[1].set_ylabel("Residuals")
        ax[1].set_title("SARIMA")

        residuals = y_test - y_pred_MLP
        residuals = [r - p for r, p in zip(y_test, y_pred_MLP)]

        ax[2].scatter(y_test, residuals)
        ax[2].axhline(
            0, color="red", linestyle="--"
        )  # Add a horizontal line at y=0 for reference
        ax[2].set_xlabel("Real Values")
        ax[2].set_ylabel("Residuals")
        ax[2].set_title("MLP")
        fig.suptitle("Residual plot")
        fig.savefig(
            rf"results_volatility\{extract_string(file, pattern)}_Residual.png", dpi=700
        )

print(f"Results are stored in {output_file}")
# %%

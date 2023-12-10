import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel


class FeatuerSelection:
    def __init__(self, df):
        self.df = df

    def __call__(self):
        y = self.df["Close"]  # Target variable
        X = self.df.drop(["Close", "Date"], axis=1)  # Drop non-numeric columns

        # Scale the data
        scaler = StandardScaler()
        x = scaler.fit_transform(X)

        # Feature selection using SVR
        svr = SVR(kernel="linear", C=0.01, shrinking=False)
        selector = SelectFromModel(svr)
        selector.fit(x, y)

        # Get selected feature indices and names
        selected_features = selector.get_support(indices=True)
        selected_feature_names = X.columns[selected_features].tolist()

        features = pd.DataFrame(selector.transform(x), columns=selected_feature_names)
        features = features.reset_index(drop=True)
        y = pd.DataFrame(y).reset_index(drop=True)
        date = self.df["Date"].reset_index(drop=True)
        df = pd.concat([date, features.join(y)], axis=1)

        # Replace 'Date' with the actual column name containing the date
        df["Date"] = pd.to_datetime(df["Date"])
        # Set the 'Date' column as the index
        df.set_index("Date", inplace=True)

        return df.sort_values("Date")

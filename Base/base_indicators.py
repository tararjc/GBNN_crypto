import pandas as pd
import numpy as np

class BaseIndicators:
    def __init__(self, df):
        self.df = df

    def _preProcess_Data(self):
        df = self.df.copy()
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values(by=["Date"], inplace=True)
        return df

    def _rsi(self, n=14):
        df = self._preProcess_Data()
        deltas = df["Close"].diff()
        seed = deltas[: n + 1]
        up = seed[seed >= 0].sum() / n
        down = -seed[seed < 0].sum() / n
        rs = up / down
        rsi = pd.Series(0.0, index=df["Close"].index)
        rsi.iloc[n] = 100.0 - 100.0 / (1.0 + rs)

        for i in range(n + 1, len(df["Close"])):
            delta = deltas[i]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta
            up = (up * (n - 1) + upval) / n
            down = (down * (n - 1) + downval) / n
            rs = up / down
            rsi.iloc[i] = 100.0 - 100.0 / (1.0 + rs)

        return rsi

    def _stoch(self, n=14, k=3, d=3):
        df = self._preProcess_Data()
        lowest_low = df["Low"].rolling(window=n, min_periods=n).min()
        highest_high = df["High"].rolling(window=n, min_periods=n).max()
        k_line = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low)

        d_line = k_line.rolling(window=k).mean()

        slow_d_line = d_line.rolling(window=d).mean()

        # return k_line, d_line, slow_d_line
        return k_line

    def _cci(self, n=20, c=0.015):
        df = self._preProcess_Data()
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        tp_ma = tp.rolling(window=n).mean()
        md = np.abs(tp - tp_ma).rolling(window=n).mean()
        cci = (tp - tp_ma) / (c * md)

        return cci

    def _adx(self, n=14):
        df = self._preProcess_Data()
        tr1 = df["High"] - df["Low"]
        tr2 = np.abs(df["High"] - df["Close"].shift())
        tr3 = np.abs(df["Low"] - df["Close"].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate the directional movement
        dm_pos = df["High"] - df["High"].shift()
        dm_neg = df["Low"].shift() - df["Low"]
        dm_pos[dm_pos < 0] = 0
        dm_neg[dm_neg < 0] = 0

        # Calculate the smoothed directional movement
        sma_pos = dm_pos.rolling(window=n).mean()
        sma_neg = dm_neg.rolling(window=n).mean()

        # Calculate the directional index (DI)
        di_pos = 100 * sma_pos / tr
        di_neg = 100 * sma_neg / tr

        # Calculate the directional index difference (DX)
        dx = 100 * np.abs(di_pos - di_neg) / (di_pos + di_neg)

        # Calculate the ADX
        adx = dx.rolling(window=n).mean()

        return adx

    def _macd(self, n_fast=12, n_slow=26, n_signal=9):
        df = self._preProcess_Data()
        ema_fast = df["Close"].ewm(span=n_fast, min_periods=n_fast).mean()
        ema_slow = df["Close"].ewm(span=n_slow, min_periods=n_slow).mean()

        # Calculate the MACD line
        macd = ema_fast - ema_slow

        # Calculate the signal line
        signal = macd.ewm(span=n_signal, min_periods=n_signal).mean()

        # Calculate the histogram
        histogram = macd - signal

        # return macd, signal, histogram
        return macd

    def _traded_value(self):
        df = self._preProcess_Data()
        return df["Volume"] * df["Close"]

    def _Volatility(self, n=14):
        df = self._preProcess_Data()
        daily_returns = (df["Close"] / df["Close"].shift(1)) - 1
        volatility = daily_returns.rolling(window=n).std() * np.sqrt(252)

        return volatility

    def _Momentum(self, n=14):
        df = self._preProcess_Data()
        momentum_period = n
        Momentum = (
            (df["Close"] - df["Close"].shift(momentum_period))
            / df["Close"].shift(momentum_period)
        ) * 100
        return Momentum

    def _ma(self, n=5):
        df = self._preProcess_Data()
        return df["Close"].rolling(window=n).mean()

    def _ema(self, n=5):
        df = self._preProcess_Data()
        return df["Close"].ewm(span=n, min_periods=n).mean()

    def _roc(self, n=2):
        df = self._preProcess_Data()
        roc = ((df["Close"] - df["Close"].shift(n)) / df["Close"].shift(n)) * 100

        return roc

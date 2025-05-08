import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from orbit.models import LGT

def is_stationary(series, alpha=0.05):
    from statsmodels.tsa.stattools import adfuller
    p = adfuller(series.dropna())[1]
    return p < alpha

def run_naive(df, targets):
    global_last_date = df['ds'].max()
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index <= global_last_date]
            pred = float(train.iloc[-1]) if len(train) > 0 else np.nan
            row[f'{metric}_pred'] = pred
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets]
    return pd.DataFrame(rows, columns=cols).set_index('Game')

def run_moving_average(df, targets, window=3):
    global_last_date = df['ds'].max()
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index <= global_last_date]
            if len(train) >= window:
                pred = float(train.iloc[-window:].mean())
            elif len(train) > 0:
                pred = float(train.mean())
            else:
                pred = np.nan
            row[f'{metric}_pred'] = pred
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets]
    return pd.DataFrame(rows, columns=cols).set_index('Game')

def run_arima(df, targets):
    global_last_date = df['ds'].max()
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index <= global_last_date]
            if len(train) < 2:
                pred = np.nan
            else:
                try:
                    model = ARIMA(train, order=(1, 1, 1))
                    res = model.fit()
                    pred = float(res.forecast(steps=1).iloc[0])
                except Exception as e:
                    print(f"ARIMA failed for {game}, {metric}: {e}")
                    pred = np.nan
            row[f'{metric}_pred'] = pred
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets]
    return pd.DataFrame(rows, columns=cols).set_index('Game')

def run_sarima(df, targets):
    global_last_date = df['ds'].max()
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index <= global_last_date]
            if len(train) < 2:
                pred = np.nan
            else:
                try:
                    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    pred = float(res.forecast(steps=1).iloc[0])
                except Exception as e:
                    print(f"SARIMA failed for {game}, {metric}: {e}")
                    pred = np.nan
            row[f'{metric}_pred'] = pred
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets]
    return pd.DataFrame(rows, columns=cols).set_index('Game')

def run_sarimax(df, targets):
    global_last_date = df['ds'].max()
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        sub['Year'] = sub.index.year
        sub['Month'] = sub.index.month
        exog = pd.DataFrame({'year': sub['Year'], 'month': sub['Month']}, index=sub.index)
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index <= global_last_date]
            exog_train = exog.loc[train.index]
            exog_test = exog.loc[[global_last_date + pd.offsets.MonthBegin(1)]]
            if len(train) < 2 or exog_train.isna().any().any() or np.isinf(exog_train).any().any():
                pred = np.nan
            else:
                try:
                    model = SARIMAX(train, exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    pred = float(res.forecast(steps=1, exog=exog_test).iloc[0])
                except Exception as e:
                    print(f"SARIMAX failed for {game}, {metric}: {e}")
                    pred = np.nan
            row[f'{metric}_pred'] = pred
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets]
    return pd.DataFrame(rows, columns=cols).set_index('Game')

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def run_lstm(df, targets, seq_length=7, epochs=20, batch_size=32):
    global_last_date = df['ds'].max()
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index <= global_last_date]
            if len(train) < seq_length + 1:
                pred = np.nan
            else:
                scaler = MinMaxScaler()
                y_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
                X_train_seq, y_train_seq = create_sequences(y_scaled, seq_length)
                if len(X_train_seq) == 0:
                    pred = np.nan
                else:
                    model = Sequential([
                        LSTM(50, activation='tanh', input_shape=(seq_length, 1), return_sequences=False),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
                    last_sequence = y_scaled[-seq_length:]
                    X_pred = np.array([last_sequence])
                    lstm_pred_scaled = model.predict(X_pred, verbose=0)
                    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
                    pred = float(lstm_pred[0, 0])
            row[f'{metric}_pred'] = pred
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets]
    return pd.DataFrame(rows, columns=cols).set_index('Game')

def run_prophet(df, targets):
    global_last_date = df['ds'].max()
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index <= global_last_date]
            if len(train) < 2:
                pred = np.nan
            else:
                prophet_df = pd.DataFrame({'ds': train.index, 'y': train.values})
                model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=1, freq='MS')
                forecast = model.predict(future)
                pred = float(forecast['yhat'].iloc[-1])
                if pred < 0 and metric in ['Avg_viewers', 'Peak_viewers', 'Hours_watched', 'Hours_streamed']:
                    pred = 0
            row[f'{metric}_pred'] = pred
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets]
    return pd.DataFrame(rows, columns=cols).set_index('Game')

def run_orbit(df, targets):
    global_last_date = df['ds'].max()
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index <= global_last_date]
            if len(train) < 2:
                pred = np.nan
            else:
                orbit_df = pd.DataFrame({'date': train.index, 'y': train.values})
                lgt = LGT(response_col='y', date_col='date', seasonality=12)
                lgt.fit(orbit_df)
                future_dates = pd.date_range(start=train.index[-1], periods=2, freq='MS')[1:]
                future_df = pd.DataFrame({'date': future_dates})
                forecast = lgt.predict(future_df)
                pred = float(forecast['prediction'].iloc[-1])
                if pred < 0 and metric in ['Avg_viewers', 'Peak_viewers', 'Hours_watched', 'Hours_streamed']:
                    pred = 0
            row[f'{metric}_pred'] = pred
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets]
    return pd.DataFrame(rows, columns=cols).set_index('Game')
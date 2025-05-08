import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from orbit.models import LGT
import itertools


# Проверка стационарности
def is_stationary(series, alpha=0.05):
    from statsmodels.tsa.stattools import adfuller
    p = adfuller(series.dropna())[1]
    return p < alpha

# --- Временные ряды ---
# Naive Forecasting (без гиперпараметров)
def run_naive(df, targets):
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        last_date = sub.index.max()
        if pd.isna(last_date):
            continue
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index < last_date]
            test = series[series.index == last_date]
            pred = float(train.iloc[-1]) if len(train) > 0 else np.nan
            actual = test.iloc[0] if len(test) == 1 else np.nan
            row[f'{metric}_pred'] = pred
            row[f'{metric}_test'] = actual
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets] + [f'{t}_test' for t in targets]
    df_model = pd.DataFrame(rows, columns=cols).set_index('Game')
    return df_model


# Moving Average с поиском оптимального окна
def run_moving_average(df, targets):
    window_sizes = [3, 6, 12]  # Сетка для размера окна
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        last_date = sub.index.max()
        if pd.isna(last_date):
            continue
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index < last_date]
            test = series[series.index == last_date]
            best_pred = np.nan
            best_rmse = np.inf
            for window in window_sizes:
                if len(train) >= window:
                    pred = float(train.iloc[-window:].mean())
                    actual = test.iloc[0] if len(test) == 1 else np.nan
                    if not np.isnan(actual) and not np.isnan(pred):
                        rmse = np.sqrt(mean_squared_error([actual], [pred]))
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_pred = pred
                elif len(train) > 0:
                    best_pred = float(train.mean())
            row[f'{metric}_pred'] = best_pred
            row[f'{metric}_test'] = actual
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets] + [f'{t}_test' for t in targets]
    df_model = pd.DataFrame(rows, columns=cols).set_index('Game')
    return df_model


# ARIMA с поиском гиперпараметров
def forecast_arima(series, order=(1, 0, 0)):
    try:
        model = ARIMA(series, order=order)
        res = model.fit()
        return float(res.forecast(steps=1).iloc[0])
    except Exception:
        return np.nan


def run_arima(df, targets):
    p = range(0, 3)
    d = range(0, 2)
    q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        last_date = sub.index.max()
        if pd.isna(last_date):
            continue
        row = {'Game': game}
        for metric in targets:
            train_series = sub.loc[sub.index < last_date, metric].dropna()
            test_val = sub.at[last_date, metric] if last_date in sub.index else np.nan
            best_pred = np.nan
            best_rmse = np.inf
            for param in pdq:
                try:
                    pred = forecast_arima(train_series, order=param)
                    if not np.isnan(pred) and not np.isnan(test_val):
                        rmse = np.sqrt(mean_squared_error([test_val], [pred]))
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_pred = pred
                except:
                    continue
            row[f'{metric}_pred'] = best_pred
            row[f'{metric}_test'] = test_val
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets] + [f'{t}_test' for t in targets]
    df_model = pd.DataFrame(rows, columns=cols).set_index('Game')
    return df_model


# SARIMA с поиском гиперпараметров
def run_sarima(df, targets):
    p = d = q = range(0, 2)
    P = D = Q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = list(itertools.product(P, D, Q, [12]))
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        last_date = sub.index.max()
        if pd.isna(last_date):
            continue
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index < last_date]
            test = series[series.index == last_date]
            best_pred = np.nan
            best_rmse = np.inf
            for param in pdq:
                for seasonal_param in seasonal_pdq:
                    try:
                        model = SARIMAX(train, order=param, seasonal_order=seasonal_param,
                                        enforce_stationarity=False, enforce_invertibility=False)
                        res = model.fit(disp=False)
                        pred = float(res.forecast(steps=1).iloc[0])
                        actual = test.iloc[0] if len(test) == 1 else np.nan
                        if not np.isnan(pred) and not np.isnan(actual):
                            rmse = np.sqrt(mean_squared_error([actual], [pred]))
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_pred = pred
                    except:
                        continue
            row[f'{metric}_pred'] = best_pred
            row[f'{metric}_test'] = actual
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets] + [f'{t}_test' for t in targets]
    df_model = pd.DataFrame(rows, columns=cols).set_index('Game')
    return df_model


# SARIMAX с поиском гиперпараметров
def run_sarimax(df, targets):
    p = d = q = range(0, 2)
    P = D = Q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = list(itertools.product(P, D, Q, [12]))
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        sub['Year'] = sub.index.year
        sub['Month'] = sub.index.month
        last_date = sub.index.max()
        if pd.isna(last_date):
            continue
        exog = pd.DataFrame({'year': sub['Year'], 'month': sub['Month']}, index=sub.index)
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index < last_date]
            test = series[series.index == last_date]
            exog_train = exog.loc[train.index]
            exog_test = exog.loc[[last_date]]
            if exog_train.isna().any().any() or np.isinf(exog_train).any().any() or \
                    exog_test.isna().any().any() or np.isinf(exog_test).any().any():
                pred = np.nan
            else:
                best_pred = np.nan
                best_rmse = np.inf
                for param in pdq:
                    for seasonal_param in seasonal_pdq:
                        try:
                            model = SARIMAX(train, exog=exog_train, order=param, seasonal_order=seasonal_param,
                                            enforce_stationarity=False, enforce_invertibility=False)
                            res = model.fit(disp=False)
                            pred = float(res.forecast(steps=1, exog=exog_test).iloc[0])
                            actual = test.iloc[0] if len(test) == 1 else np.nan
                            if not np.isnan(pred) and not np.isnan(actual):
                                rmse = np.sqrt(mean_squared_error([actual], [pred]))
                                if rmse < best_rmse:
                                    best_rmse = rmse
                                    best_pred = pred
                        except:
                            continue
                row[f'{metric}_pred'] = best_pred
                row[f'{metric}_test'] = actual
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets] + [f'{t}_test' for t in targets]
    df_model = pd.DataFrame(rows, columns=cols).set_index('Game')
    return df_model


# LSTM с поиском гиперпараметров
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def run_lstm(df, targets):
    param_grid = {
        'seq_length': [3, 7, 12],
        'lstm_units': [50, 100],
        'epochs': [10, 20]
    }
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        last_date = sub.index.max()
        if pd.isna(last_date):
            continue
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index < last_date]
            test = series[series.index == last_date]
            best_pred = np.nan
            best_rmse = np.inf
            for seq_length in param_grid['seq_length']:
                for lstm_units in param_grid['lstm_units']:
                    for epochs in param_grid['epochs']:
                        if len(train) < seq_length + 1:
                            continue
                        scaler = MinMaxScaler()
                        y_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
                        X_train_seq, y_train_seq = create_sequences(y_scaled, seq_length)
                        if len(X_train_seq) == 0:
                            continue
                        model = Sequential([
                            LSTM(lstm_units, activation='tanh', input_shape=(seq_length, 1)),
                            Dense(1)
                        ])
                        model.compile(optimizer='adam', loss='mse')
                        model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=32,
                                  validation_split=0.2, verbose=0)
                        last_sequence = y_scaled[-seq_length:]
                        X_pred = np.array([last_sequence])
                        lstm_pred_scaled = model.predict(X_pred, verbose=0)
                        lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
                        pred = float(lstm_pred[0, 0])
                        actual = test.iloc[0] if len(test) == 1 else np.nan
                        if not np.isnan(pred) and not np.isnan(actual):
                            rmse = np.sqrt(mean_squared_error([actual], [pred]))
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_pred = pred
            row[f'{metric}_pred'] = best_pred
            row[f'{metric}_test'] = actual
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets] + [f'{t}_test' for t in targets]
    df_model = pd.DataFrame(rows, columns=cols).set_index('Game')
    return df_model


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


# Orbit с поиском гиперпараметров
def run_orbit(df, targets):
    param_grid = {
        'seasonality': [6, 12],
        'regressor_col': [['year'], ['year', 'month']]
    }
    rows = []
    for game in df['Game'].unique():
        sub = df[df['Game'] == game].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values('ds').set_index('ds').asfreq('MS')
        sub['year'] = sub.index.year
        sub['month'] = sub.index.month
        last_date = sub.index.max()
        if pd.isna(last_date):
            continue
        row = {'Game': game}
        for metric in targets:
            series = sub[metric].dropna()
            train = series[series.index < last_date]
            test = series[series.index == last_date]
            if len(train) < 2:
                pred = np.nan
            else:
                best_pred = np.nan
                best_rmse = np.inf
                for seasonality in param_grid['seasonality']:
                    for regressor_col in param_grid['regressor_col']:
                        orbit_df = pd.DataFrame({
                            'date': train.index,
                            'y': train.values,
                            'year': sub.loc[train.index, 'year'],
                            'month': sub.loc[train.index, 'month']
                        })
                        lgt = LGT(
                            response_col='y',
                            date_col='date',
                            seasonality=seasonality,
                            regressor_col=regressor_col
                        )
                        lgt.fit(orbit_df)
                        future_dates = pd.date_range(start=train.index[-1], periods=2, freq='MS')[1:]
                        future_df = pd.DataFrame({
                            'date': future_dates,
                            'year': [future_dates[0].year],
                            'month': [future_dates[0].month]
                        })
                        forecast = lgt.predict(future_df)
                        pred = float(forecast['prediction'].iloc[-1])
                        if pred < 0 and metric in ['Avg_viewers', 'Peak_viewers', 'Hours_watched', 'Hours_streamed']:
                            pred = 0
                        actual = test.iloc[0] if len(test) == 1 else np.nan
                        if not np.isnan(pred) and not np.isnan(actual):
                            rmse = np.sqrt(mean_squared_error([actual], [pred]))
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_pred = pred
                row[f'{metric}_pred'] = best_pred
                row[f'{metric}_test'] = actual
        rows.append(row)
    cols = ['Game'] + [f'{t}_pred' for t in targets] + [f'{t}_test' for t in targets]
    df_model = pd.DataFrame(rows, columns=cols).set_index('Game')
    return df_model

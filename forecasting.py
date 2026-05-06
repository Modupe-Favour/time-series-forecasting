import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Load & Prepare Data ──────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    df.columns = ["Month", "Passengers"]
    df["Month"] = pd.to_datetime(df["Month"])
    df.set_index("Month", inplace=True)
    df.sort_index(inplace=True)
    return df

def train_test_split_ts(df, test_months=24):
    """Split time series into train and test sets"""
    train = df.iloc[:-test_months]
    test = df.iloc[-test_months:]
    return train, test

# ── Evaluation Metrics ───────────────────────────────────────
def evaluate_model(actual, predicted, model_name):
    """Calculate MAE, RMSE, MAPE"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2)
    }

# ── MODEL 1: ARIMA ───────────────────────────────────────────
def run_arima(train, test, forecast_steps=24):
    from statsmodels.tsa.arima.model import ARIMA
    print("🔄 Training ARIMA...")

    # Fit ARIMA(2,1,2)
    model = ARIMA(train["Passengers"], order=(2, 1, 2))
    fitted = model.fit()

    # Forecast
    forecast_obj = fitted.get_forecast(steps=forecast_steps)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    forecast.index = test.index
    conf_int.index = test.index

    metrics = evaluate_model(test["Passengers"],
                              forecast, "ARIMA")
    print(f"✅ ARIMA — MAPE: {metrics['MAPE (%)']:.2f}%")

    return {
        "model_name": "ARIMA(2,1,2)",
        "forecast": forecast,
        "conf_int": conf_int,
        "metrics": metrics,
        "fitted": fitted
    }

# ── MODEL 2: SARIMA ──────────────────────────────────────────
def run_sarima(train, test, forecast_steps=24):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    print("🔄 Training SARIMA...")

    # SARIMA(1,1,1)(1,1,1,12) — seasonal period = 12 months
    model = SARIMAX(train["Passengers"],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fitted = model.fit(disp=False)

    forecast_obj = fitted.get_forecast(steps=forecast_steps)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    forecast.index = test.index
    conf_int.index = test.index

    metrics = evaluate_model(test["Passengers"],
                              forecast, "SARIMA")
    print(f"✅ SARIMA — MAPE: {metrics['MAPE (%)']:.2f}%")

    return {
        "model_name": "SARIMA(1,1,1)(1,1,1,12)",
        "forecast": forecast,
        "conf_int": conf_int,
        "metrics": metrics,
        "fitted": fitted
    }

# ── MODEL 3: PROPHET ─────────────────────────────────────────
def run_prophet(train, test, forecast_steps=24):
    from prophet import Prophet
    print("🔄 Training Prophet...")

    # Prophet requires columns named ds and y
    train_prophet = train.reset_index()
    train_prophet.columns = ["ds", "y"]

    model = Prophet(yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode="multiplicative",
                    interval_width=0.95)
    model.fit(train_prophet)

    # Create future dataframe
    future = model.make_future_dataframe(
        periods=forecast_steps, freq="MS")
    forecast_df = model.predict(future)

    # Extract forecast for test period
    forecast = forecast_df.set_index("ds")["yhat"].iloc[-forecast_steps:]
    forecast.index = test.index

    conf_int = pd.DataFrame({
        "lower Passengers": forecast_df.set_index(
            "ds")["yhat_lower"].iloc[-forecast_steps:].values,
        "upper Passengers": forecast_df.set_index(
            "ds")["yhat_upper"].iloc[-forecast_steps:].values
    }, index=test.index)

    metrics = evaluate_model(test["Passengers"],
                              forecast, "Prophet")
    print(f"✅ Prophet — MAPE: {metrics['MAPE (%)']:.2f}%")

    return {
        "model_name": "Facebook Prophet",
        "forecast": forecast,
        "conf_int": conf_int,
        "metrics": metrics,
        "model": model,
        "forecast_df": forecast_df
    }

# ── MODEL 4: LSTM ────────────────────────────────────────────
def run_lstm(train, test, forecast_steps=24):
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    print("🔄 Training LSTM...")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(
        train["Passengers"].values.reshape(-1, 1))

    # Create sequences
    SEQ_LEN = 12

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled, SEQ_LEN)

    # Build LSTM model
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                              input_shape=(SEQ_LEN, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train,
              epochs=50, batch_size=16, verbose=0,
              validation_split=0.1)

    # Forecast step by step
    last_sequence = train_scaled[-SEQ_LEN:]
    predictions = []

    for _ in range(forecast_steps):
        pred = model.predict(
            last_sequence.reshape(1, SEQ_LEN, 1), verbose=0)
        predictions.append(pred[0, 0])
        last_sequence = np.append(
            last_sequence[1:], pred).reshape(-1, 1)

    # Inverse transform
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)).flatten()

    forecast = pd.Series(predictions, index=test.index)

    # Simple confidence interval (±10% for LSTM)
    conf_int = pd.DataFrame({
        "lower Passengers": forecast * 0.90,
        "upper Passengers": forecast * 1.10
    }, index=test.index)

    metrics = evaluate_model(test["Passengers"],
                              forecast, "LSTM")
    print(f"✅ LSTM — MAPE: {metrics['MAPE (%)']:.2f}%")

    return {
        "model_name": "LSTM Neural Network",
        "forecast": forecast,
        "conf_int": conf_int,
        "metrics": metrics
    }

# ── Run All Models ───────────────────────────────────────────
def run_all_models(data_path="Data/air_passengers.csv",
                   test_months=24):
    df = load_data(data_path)
    train, test = train_test_split_ts(df, test_months)

    print(f"Training samples: {len(train)}")
    print(f"Test samples: {len(test)}\n")

    results = {}
    results["arima"] = run_arima(train, test, test_months)
    results["sarima"] = run_sarima(train, test, test_months)
    results["prophet"] = run_prophet(train, test, test_months)
    results["lstm"] = run_lstm(train, test, test_months)

    # Save metrics comparison
    metrics_df = pd.DataFrame([
        results["arima"]["metrics"],
        results["sarima"]["metrics"],
        results["prophet"]["metrics"],
        results["lstm"]["metrics"]
    ])
    metrics_df.to_csv("model_results/metrics_comparison.csv",
                      index=False)
    print("\n✅ Metrics saved to model_results/metrics_comparison.csv")

    # Save forecasts
    for key, res in results.items():
        forecast_df = pd.DataFrame({
            "Date": res["forecast"].index,
            "Forecast": res["forecast"].values,
            "Lower": res["conf_int"].iloc[:, 0].values,
            "Upper": res["conf_int"].iloc[:, 1].values
        })
        forecast_df.to_csv(
            f"model_results/{key}_forecast.csv", index=False)

    print("✅ All forecasts saved!")
    return df, train, test, results
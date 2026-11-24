import json
import logging
import os
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.etl_fetch import fetch_btc_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
RAW_DATA_PATH = os.path.join(DATA_DIR, "btc_raw.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "btc_features.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "btc_forecaster.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

FEATURE_COLS = [
    "close_lag_1",
    "close_lag_2",
    "close_lag_3",
    "close_lag_7",
    "ma_7",
    "ma_30",
    "volatility_7",
    "return_1d",
]


def engineer_features(
    raw_path: str = RAW_DATA_PATH, out_path: str = PROCESSED_DATA_PATH
) -> pd.DataFrame:
    """Clean BTC data and compute moving averages, lags, and volatility."""
    if not os.path.exists(raw_path):
        fetch_btc_data(raw_path)

    df = pd.read_csv(raw_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)

    df["close_lag_1"] = df["Close"].shift(1)
    df["close_lag_2"] = df["Close"].shift(2)
    df["close_lag_3"] = df["Close"].shift(3)
    df["close_lag_7"] = df["Close"].shift(7)
    df["ma_7"] = df["Close"].rolling(window=7).mean()
    df["ma_30"] = df["Close"].rolling(window=30).mean()
    df["return_1d"] = df["Close"].pct_change()
    df["volatility_7"] = df["return_1d"].rolling(window=7).std()
    df["target_next_close"] = df["Close"].shift(-1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved engineered features (%d rows) to %s", len(df), out_path)
    return df


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(
        np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
    )


def train_and_evaluate(
    features_path: str = PROCESSED_DATA_PATH,
    model_dir: str = MODEL_DIR,
    mape_threshold: float = 25.0,
) -> dict:
    """Train Ridge regression model, evaluate on last 7 days, and save artifacts."""
    if not os.path.exists(features_path):
        engineer_features()

    df = pd.read_csv(features_path)
    trainable = df.dropna(subset=FEATURE_COLS + ["target_next_close"]).copy()

    if len(trainable) < 30:
        raise ValueError("Not enough rows to train model.")

    train_df = trainable.iloc[:-7]
    test_df = trainable.iloc[-7:]

    model = Ridge(alpha=1.0)
    model.fit(train_df[FEATURE_COLS], train_df["target_next_close"])

    preds = model.predict(test_df[FEATURE_COLS])
    y_true = test_df["target_next_close"].values

    mae = float(mean_absolute_error(y_true, preds))
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    mape = compute_mape(y_true, preds)

    # Refit on all available trainable data for production forecasting
    model.fit(trainable[FEATURE_COLS], trainable["target_next_close"])

    recent_closes = df["Close"].dropna().tail(35).tolist()
    last_date = str(df["Date"].iloc[-1])[:10]
    version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")

    metrics = {
        "model_version": version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_data_date": last_date,
        "test_window_days": 7,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 4),
        "passed_threshold": mape <= mape_threshold,
    }

    if mape <= mape_threshold:
        os.makedirs(model_dir, exist_ok=True)
        artifact = {
            "model": model,
            "feature_cols": FEATURE_COLS,
            "recent_closes": recent_closes,
            "last_data_date": last_date,
            "model_version": version,
        }
        joblib.dump(artifact, os.path.join(model_dir, "btc_forecaster.joblib"))
        with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Registered model %s | Metrics: %s", version, metrics)
    else:
        logger.warning(
            "Model MAPE %.2f%% exceeded threshold %.2f%%", mape, mape_threshold
        )

    return metrics


def recursive_forecast(
    model, history_closes: list, forecast_days: int = 7, start_date: str = None
) -> list:
    """Forecast next N days recursively using recent closing prices."""
    closes = [float(x) for x in history_closes]
    if len(closes) < 30:
        closes = [closes[0]] * (30 - len(closes)) + closes

    base_date = (
        pd.to_datetime(start_date)
        if start_date
        else pd.Timestamp.utcnow().normalize()
    )

    predictions = []
    for step in range(1, forecast_days + 1):
        series = pd.Series(closes)
        returns = series.pct_change()
        row = pd.DataFrame(
            [
                {
                    "close_lag_1": closes[-1],
                    "close_lag_2": closes[-2],
                    "close_lag_3": closes[-3],
                    "close_lag_7": closes[-7],
                    "ma_7": float(series.tail(7).mean()),
                    "ma_30": float(series.tail(30).mean()),
                    "volatility_7": float(returns.tail(7).std() or 0.0),
                    "return_1d": float(returns.iloc[-1] or 0.0),
                }
            ]
        )
        next_price = float(model.predict(row[FEATURE_COLS])[0])
        next_price = max(next_price, 0.0)
        closes.append(next_price)
        pred_date = (base_date + pd.Timedelta(days=step)).strftime("%Y-%m-%d")
        predictions.append({"date": pred_date, "predicted_close": round(next_price, 2)})

    return predictions


if __name__ == "__main__":
    fetch_btc_data()
    engineer_features()
    train_and_evaluate()
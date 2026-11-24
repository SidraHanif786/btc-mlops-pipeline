from kfp import compiler, dsl


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "yfinance", "requests"],
)
def ingest_btc_data(raw_data: dsl.Output[dsl.Dataset]):
    import yfinance as yf
    import pandas as pd

    df = yf.download("BTC-USD", period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.dropna().to_csv(raw_data.path, index=False)


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy"],
)
def generate_features(
    raw_data: dsl.Input[dsl.Dataset],
    features_data: dsl.Output[dsl.Dataset],
):
    import pandas as pd

    df = pd.read_csv(raw_data.path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["close_lag_1"] = df["Close"].shift(1)
    df["close_lag_2"] = df["Close"].shift(2)
    df["close_lag_3"] = df["Close"].shift(3)
    df["close_lag_7"] = df["Close"].shift(7)
    df["ma_7"] = df["Close"].rolling(7).mean()
    df["ma_30"] = df["Close"].rolling(30).mean()
    df["return_1d"] = df["Close"].pct_change()
    df["volatility_7"] = df["return_1d"].rolling(7).std()
    df["target_next_close"] = df["Close"].shift(-1)
    df.dropna().to_csv(features_data.path, index=False)


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"],
)
def train_btc_model(
    features_data: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Output[dsl.Model],
    metrics_artifact: dsl.Output[dsl.Metrics],
):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    cols = [
        "close_lag_1",
        "close_lag_2",
        "close_lag_3",
        "close_lag_7",
        "ma_7",
        "ma_30",
        "volatility_7",
        "return_1d",
    ]
    df = pd.read_csv(features_data.path)
    train_df, test_df = df.iloc[:-7], df.iloc[-7:]

    model = Ridge(alpha=1.0)
    model.fit(train_df[cols], train_df["target_next_close"])
    preds = model.predict(test_df[cols])
    y_true = test_df["target_next_close"].values

    mae = float(mean_absolute_error(y_true, preds))
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    mape = float(np.mean(np.abs((y_true - preds) / np.clip(y_true, 1e-8, None))) * 100)

    metrics_artifact.log_metric("MAE", mae)
    metrics_artifact.log_metric("RMSE", rmse)
    metrics_artifact.log_metric("MAPE", mape)
    joblib.dump(model, model_artifact.path)


@dsl.pipeline(
    name="btc-7day-forecast-pipeline",
    description="Automated BTC-USD ingestion, feature engineering, and model training.",
)
def btc_forecast_pipeline():
    ingest_task = ingest_btc_data()
    feature_task = generate_features(raw_data=ingest_task.outputs["raw_data"])
    train_btc_model(features_data=feature_task.outputs["features_data"])


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=btc_forecast_pipeline,
        package_path="kubeflow/btc_kfp_pipeline.yaml",
    )
    print("Compiled kubeflow/btc_kfp_pipeline.yaml successfully!")

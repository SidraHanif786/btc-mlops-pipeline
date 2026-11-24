import json
import logging
import os
import time
import joblib
import numpy as np
from flask import Flask, jsonify, request
from sklearn.linear_model import Ridge
from src.train_forecast import FEATURE_COLS, recursive_forecast

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
RELOAD_TOKEN = os.getenv("RELOAD_TOKEN", "dev-reload-token")

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

app = Flask(__name__)
START_TIME = time.time()

STATE = {
    "artifact": None,
    "metrics": {},
}


def build_fallback_artifact() -> dict:
    """Create a lightweight fallback model if PVC is empty on first boot."""
    rng = np.random.default_rng(42)
    base_prices = 65000 + np.cumsum(rng.normal(0, 300, size=60))
    X, y = [], []
    for i in range(30, len(base_prices) - 1):
        window = base_prices[: i + 1]
        rets = np.diff(window) / window[:-1]
        X.append(
            [
                window[-1],
                window[-2],
                window[-3],
                window[-7],
                float(np.mean(window[-7:])),
                float(np.mean(window[-30:])),
                float(np.std(rets[-7:])),
                float(rets[-1]),
            ]
        )
        y.append(base_prices[i + 1])
    model = Ridge(alpha=1.0).fit(np.array(X), np.array(y))
    return {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "recent_closes": base_prices[-35:].tolist(),
        "last_data_date": time.strftime("%Y-%m-%d"),
        "model_version": "v0-fallback",
    }


def load_model_artifacts() -> dict:
    """Load serialized model and metrics from MODEL_DIR."""
    model_path = os.path.join(MODEL_DIR, "btc_forecaster.joblib")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")

    if os.path.exists(model_path):
        STATE["artifact"] = joblib.load(model_path)
        logger.info("Loaded model %s", STATE["artifact"].get("model_version"))
    else:
        logger.warning("Model file not found at %s; using fallback.", model_path)
        STATE["artifact"] = build_fallback_artifact()

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            STATE["metrics"] = json.load(f)
    else:
        STATE["metrics"] = {
            "model_version": STATE["artifact"]["model_version"],
            "MAE": 0.0,
            "RMSE": 0.0,
            "MAPE": 0.0,
            "note": "Fallback metrics until ETL pipeline runs",
        }
    return STATE["artifact"]


load_model_artifacts()


@app.route("/health", methods=["GET"])
def health():
    artifact = STATE["artifact"] or load_model_artifacts()
    return (
        jsonify(
            {
                "status": "ok",
                "stage": MODEL_STAGE,
                "uptime_seconds": round(time.time() - START_TIME, 2),
                "model_version": artifact.get("model_version", "unknown"),
            }
        ),
        200,
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify(STATE["metrics"]), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON payload must be an object."}), 400

        forecast_days = payload.get("forecast_days", 7)
        if (
            not isinstance(forecast_days, int)
            or isinstance(forecast_days, bool)
            or forecast_days < 1
            or forecast_days > 30
        ):
            return (
                jsonify(
                    {"error": "forecast_days must be an integer between 1 and 30."}
                ),
                400,
            )

        artifact = STATE["artifact"] or load_model_artifacts()
        custom_history = payload.get("historical_closes")

        if custom_history is not None:
            if (
                not isinstance(custom_history, list)
                or len(custom_history) < 7
                or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
                    for v in custom_history
                )
            ):
                return (
                    jsonify(
                        {
                            "error": (
                                "historical_closes must be a list of at least "
                                "7 positive numbers."
                            )
                        }
                    ),
                    400,
                )
            history = custom_history
        else:
            history = artifact["recent_closes"]

        preds = recursive_forecast(
            model=artifact["model"],
            history_closes=history,
            forecast_days=forecast_days,
            start_date=artifact.get("last_data_date"),
        )
        return (
            jsonify(
                {
                    "model_version": artifact.get("model_version"),
                    "forecast_days": forecast_days,
                    "predictions": preds,
                }
            ),
            200,
        )
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return jsonify({"error": "Internal prediction error."}), 500


@app.route("/reload", methods=["POST"])
def reload_model():
    token = request.headers.get("X-Reload-Token", "")
    if RELOAD_TOKEN and token != RELOAD_TOKEN:
        return jsonify({"error": "Unauthorized reload request."}), 401
    try:
        artifact = load_model_artifacts()
        return (
            jsonify(
                {
                    "status": "reloaded",
                    "model_version": artifact.get("model_version"),
                }
            ),
            200,
        )
    except Exception as exc:
        logger.exception("Reload failed: %s", exc)
        return jsonify({"error": "Failed to reload model."}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)

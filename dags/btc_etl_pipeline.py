from datetime import datetime, timedelta
import os
import sys
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/project")
sys.path.append("/opt/airflow/venv_packages")

FLASK_RELOAD_URL = os.getenv("FLASK_RELOAD_URL", "http://flask-api:5000/reload")
RELOAD_TOKEN = os.getenv("RELOAD_TOKEN", "dev-reload-token")
DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/shared_data")
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/shared_models")


def task_fetch_data():
    from src.etl_fetch import fetch_btc_data

    raw_path = os.path.join(DATA_DIR, "btc_raw.csv")
    fetch_btc_data(output_path=raw_path)


def task_preprocess_features():
    from src.train_forecast import engineer_features

    raw_path = os.path.join(DATA_DIR, "btc_raw.csv")
    feat_path = os.path.join(DATA_DIR, "btc_features.csv")
    engineer_features(raw_path=raw_path, out_path=feat_path)


def task_retrain_and_evaluate():
    from src.train_forecast import train_and_evaluate

    feat_path = os.path.join(DATA_DIR, "btc_features.csv")
    metrics = train_and_evaluate(features_path=feat_path, model_dir=MODEL_DIR)
    if not metrics.get("passed_threshold"):
        raise ValueError(f"Model failed threshold check: {metrics}")


def task_trigger_api_reload():
    resp = requests.post(
        FLASK_RELOAD_URL,
        headers={"X-Reload-Token": RELOAD_TOKEN},
        timeout=15,
    )
    resp.raise_for_status()
    print("Flask API reload response:", resp.json())


default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="btc_daily_etl_and_retrain",
    default_args=default_args,
    description="Daily BTC-USD ETL, feature engineering, retraining, and API reload",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "btc", "forecast"],
) as dag:
    fetch_task = PythonOperator(
        task_id="fetch_btc_data",
        python_callable=task_fetch_data,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_and_feature_engineering",
        python_callable=task_preprocess_features,
    )

    retrain_task = PythonOperator(
        task_id="retrain_and_evaluate_model",
        python_callable=task_retrain_and_evaluate,
    )

    reload_task = PythonOperator(
        task_id="trigger_flask_model_reload",
        python_callable=task_trigger_api_reload,
    )

    fetch_task >> preprocess_task >> retrain_task >> reload_task

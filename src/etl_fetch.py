import os
import logging
from datetime import datetime
import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "btc_raw.csv")


def fetch_from_yfinance(period: str = "2y") -> pd.DataFrame:
    """Download daily BTC-USD OHLCV data from Yahoo Finance."""
    logger.info("Fetching BTC-USD data from Yahoo Finance (%s)...", period)
    df = yf.download("BTC-USD", period=period, interval="1d", progress=False)
    if df is None or df.empty:
        raise ValueError("Empty dataframe returned from yfinance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df


def fetch_from_binance(limit: int = 500) -> pd.DataFrame:
    """Fallback: fetch daily BTCUSDT klines from Binance Public API."""
    logger.info("Fetching BTCUSDT data from Binance Public API...")
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1d", "limit": limit}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    rows = []
    for k in resp.json():
        rows.append(
            {
                "Date": datetime.utcfromtimestamp(k[0] / 1000).strftime("%Y-%m-%d"),
                "Open": float(k[1]),
                "High": float(k[2]),
                "Low": float(k[3]),
                "Close": float(k[4]),
                "Volume": float(k[5]),
            }
        )
    return pd.DataFrame(rows)


def fetch_btc_data(output_path: str = RAW_DATA_PATH) -> str:
    """Fetch BTC daily market data with automatic fallback and save to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        df = fetch_from_yfinance()
    except Exception as exc:
        logger.warning("yfinance failed (%s); falling back to Binance API.", exc)
        df = fetch_from_binance()

    df = df.dropna().sort_values("Date").drop_duplicates(subset=["Date"])
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return output_path


if __name__ == "__main__":
    fetch_btc_data()
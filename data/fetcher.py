# fetcher.py
# ----------
# Pulls OHLCV data from yfinance and caches it in memory.
#
# GCC indices (ADX, TASI, DFM) are sometimes flaky on yfinance —
# data gaps, delisting, wrong symbols. So if a ticker fails we
# fall back to AAPL/MSFT/AMZN/NVDA to keep the dashboard working.
#
# Cache is a simple dict keyed by (ticker, period). Good enough
# to avoid hammering the API on every Streamlit rerun.

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

_CACHE: dict[str, pd.DataFrame] = {}

GCC_TICKERS = {
    "ADX (Abu Dhabi)":  "^FTFADGI",
    "TADAWUL (Saudi)":  "^TASI",
    "DFM (Dubai)":      "^DFMGI",
    "Apple":            "AAPL",
    "Microsoft":        "MSFT",
    "Amazon":           "AMZN",
    "NVIDIA":           "NVDA",
    "Meta":             "META",
}

FALLBACK_TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA"]


def fetch_ohlcv(ticker: str, period_years: int = 3, use_cache: bool = True) -> pd.DataFrame:
    cache_key = f"{ticker}_{period_years}y"
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    end   = datetime.today()
    start = end - timedelta(days=365 * period_years)
    df    = _download(ticker, start, end)

    if df is None or df.empty:
        for fb in FALLBACK_TICKERS:
            df = _download(fb, start, end)
            if df is not None and not df.empty:
                break

    if df is None or df.empty:
        raise ValueError(f"No data for {ticker} or fallbacks")

    df = _clean(df)
    _CACHE[cache_key] = df
    return df


def fetch_multiple(tickers: list[str], period_years: int = 3) -> dict[str, pd.DataFrame]:
    result = {}
    for t in tickers:
        try:
            result[t] = fetch_ohlcv(t, period_years=period_years)
        except Exception:
            pass
    return result


def get_latest_price(ticker: str) -> float:
    return float(fetch_ohlcv(ticker, period_years=1)["Close"].iloc[-1])


def _download(ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        return df if not df.empty else None
    except Exception:
        return None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.dropna(subset=["Close"], inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df["Returns"]     = df["Close"].pct_change()
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df

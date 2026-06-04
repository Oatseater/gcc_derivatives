"""yfinance data fetcher with caching for GCC indices."""
import os
import time
import pandas as pd
import yfinance as yf

GCC_TICKERS = {
    "ADX (Abu Dhabi)": "^FTFADGI",
    "TADAWUL (Saudi)": "^TASI",
    "DFM (Dubai)": "^DFMGI",
    "AAPL (fallback)": "AAPL",
    "MSFT (fallback)": "MSFT",
}

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
_TTL = 3600  # seconds


def _cache_path(key):
    safe = "".join(c if c.isalnum() else "_" for c in key)
    return os.path.join(_CACHE_DIR, f"{safe}.pkl")


def fetch(ticker, period="2y", interval="1d"):
    """Fetch OHLCV for one ticker, cached. Returns clean DataFrame, date index."""
    key = f"{ticker}_{period}_{interval}"
    cp = _cache_path(key)
    if os.path.exists(cp) and (time.time() - os.path.getmtime(cp) < _TTL):
        try:
            return pd.read_pickle(cp)
        except Exception:
            pass
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    try:
        df.to_pickle(cp)
    except Exception:
        pass
    return df


def fetch_close_panel(tickers, period="2y"):
    """Return DataFrame of aligned Close prices for multiple tickers."""
    cols = {}
    for label, tk in tickers.items():
        d = fetch(tk, period=period)
        if not d.empty:
            cols[label] = d["Close"]
    if not cols:
        return pd.DataFrame()
    panel = pd.concat(cols, axis=1).dropna()
    return panel


def get_returns(ticker, period="2y"):
    df = fetch(ticker, period=period)
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"].pct_change().dropna()

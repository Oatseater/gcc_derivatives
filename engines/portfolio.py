"""Portfolio optimization: efficient frontier, max Sharpe, min variance."""
import numpy as np
import pandas as pd


def returns_cov(prices, rf=0.04):
    """Daily->annual mean returns and covariance."""
    rets = prices.pct_change().dropna()
    mean_ann = rets.mean() * 252
    cov_ann = rets.cov() * 252
    return rets, mean_ann, cov_ann


def efficient_frontier(mean_ann, cov_ann, rf=0.04, n_port=500, seed=42):
    """Generate n_port random portfolios."""
    np.random.seed(seed)
    n = len(mean_ann)
    mu = mean_ann.values
    cov = cov_ann.values
    res = {"ret": [], "vol": [], "sharpe": [], "weights": []}
    for _ in range(n_port):
        w = np.random.random(n)
        w /= w.sum()
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov @ w))
        s = (r - rf) / v if v > 0 else 0.0
        res["ret"].append(r)
        res["vol"].append(v)
        res["sharpe"].append(s)
        res["weights"].append(w)
    for k in ("ret", "vol", "sharpe"):
        res[k] = np.array(res[k])
    res["weights"] = np.array(res["weights"])
    return res


def max_sharpe(frontier, tickers):
    i = int(np.argmax(frontier["sharpe"]))
    return _pack(frontier, i, tickers)


def min_variance(frontier, tickers):
    i = int(np.argmin(frontier["vol"]))
    return _pack(frontier, i, tickers)


def _pack(frontier, i, tickers):
    return {"ret": float(frontier["ret"][i]), "vol": float(frontier["vol"][i]),
            "sharpe": float(frontier["sharpe"][i]),
            "weights": dict(zip(tickers, frontier["weights"][i].round(4)))}


def analyze(prices, rf=0.04, n_port=500):
    rets, mean_ann, cov_ann = returns_cov(prices, rf)
    fr = efficient_frontier(mean_ann, cov_ann, rf, n_port)
    tickers = list(prices.columns)
    return {"frontier": fr, "max_sharpe": max_sharpe(fr, tickers),
            "min_var": min_variance(fr, tickers),
            "mean_ann": mean_ann, "cov_ann": cov_ann, "tickers": tickers}

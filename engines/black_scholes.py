"""Black-Scholes pricer + Greeks + implied vol. NumPy only."""
import numpy as np


def _norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(_erf)(x / np.sqrt(2.0)))


def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _erf(x):
    # Abramowitz-Stegun 7.1.26
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741)
               * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return np.sign(x) * y


def _d1_d2(S, K, T, r, sigma):
    S, K, T, r, sigma = map(float, (S, K, T, r, sigma))
    T = max(T, 1e-12)
    sigma = max(sigma, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def price(S, K, T, r, sigma, kind="call"):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if kind == "call":
        return S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
    return K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def greeks(S, K, T, r, sigma, kind="call"):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    T = max(float(T), 1e-12)
    sqT = np.sqrt(T)
    pdf = _norm_pdf(d1)
    disc = np.exp(-r * T)

    delta = _norm_cdf(d1) if kind == "call" else _norm_cdf(d1) - 1.0
    gamma = pdf / (S * sigma * sqT)
    vega = S * pdf * sqT / 100.0  # per 1% vol
    if kind == "call":
        theta = (-S * pdf * sigma / (2 * sqT) - r * K * disc * _norm_cdf(d2)) / 365.0
        rho = K * T * disc * _norm_cdf(d2) / 100.0
    else:
        theta = (-S * pdf * sigma / (2 * sqT) + r * K * disc * _norm_cdf(-d2)) / 365.0
        rho = -K * T * disc * _norm_cdf(-d2) / 100.0

    return {"Delta": float(delta), "Gamma": float(gamma), "Theta": float(theta),
            "Vega": float(vega), "Rho": float(rho)}


def implied_vol(target, S, K, T, r, kind="call", tol=1e-6, max_iter=100):
    """Newton-Raphson IV solver."""
    sigma = 0.25
    for _ in range(max_iter):
        p = price(S, K, T, r, sigma, kind)
        v = greeks(S, K, T, r, sigma, kind)["Vega"] * 100.0  # back to per-1.0
        if abs(v) < 1e-10:
            break
        diff = p - target
        if abs(diff) < tol:
            return float(sigma)
        sigma -= diff / v
        sigma = max(sigma, 1e-6)
    return float(sigma)

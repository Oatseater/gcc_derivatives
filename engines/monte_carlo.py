"""Monte Carlo GBM simulation, VaR/CVaR, stress tests."""
import numpy as np


def simulate_paths(S0, mu, sigma, T, steps=252, N=10000, seed=None):
    """Geometric Brownian Motion paths. Returns array (N, steps+1)."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / steps
    Z = np.random.standard_normal((N, steps))
    incr = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    logpaths = np.cumsum(incr, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((N, 1)), logpaths]))
    return paths


def var_cvar(paths, S0):
    """VaR & CVaR at 95% and 99% on terminal P&L."""
    terminal = paths[:, -1]
    pnl = terminal - S0
    out = {}
    for c in (0.95, 0.99):
        var = -np.percentile(pnl, (1 - c) * 100)
        tail = pnl[pnl <= -var]
        cvar = -tail.mean() if tail.size else var
        out[f"VaR_{int(c*100)}"] = float(var)
        out[f"CVaR_{int(c*100)}"] = float(cvar)
    return out


def stress_test(S0):
    """Apply historical crash shocks."""
    scenarios = {"2008 Crash": -0.40, "COVID-19": -0.30, "Gulf Crisis": -0.20}
    return {k: {"shock": v, "price": float(S0 * (1 + v)),
                "loss": float(S0 * v)} for k, v in scenarios.items()}


def summary(S0, mu, sigma, T, N=10000, steps=252, seed=42):
    paths = simulate_paths(S0, mu, sigma, T, steps, N, seed)
    risk = var_cvar(paths, S0)
    return {"paths": paths, "risk": risk, "stress": stress_test(S0),
            "terminal_mean": float(paths[:, -1].mean()),
            "terminal_std": float(paths[:, -1].std())}

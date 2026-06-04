"""GARCH(1,1) volatility modeling and forecasting."""
import numpy as np
import pandas as pd
from arch import arch_model


def historical_vol(returns, annualize=True):
    """Annualized historical volatility from daily returns."""
    vol = np.std(returns) * (np.sqrt(252) if annualize else 1.0)
    return float(vol)


def fit_garch(returns):
    """Fit GARCH(1,1). returns: daily simple returns (decimal)."""
    r = pd.Series(returns).dropna() * 100.0  # arch prefers %-scaled
    am = arch_model(r, vol="GARCH", p=1, q=1, mean="Constant", dist="normal")
    res = am.fit(disp="off")
    return res


def forecast_vol(res, horizon=30):
    """Forecast daily vol (annualized %) for `horizon` days."""
    fc = res.forecast(horizon=horizon, reindex=False)
    var_daily = fc.variance.values[-1]            # %^2 daily
    daily_vol = np.sqrt(var_daily)                # % daily
    ann_vol = daily_vol * np.sqrt(252)            # % annualized
    return ann_vol  # array length=horizon, in percent


def classify_regime(ann_vol_pct):
    """calm / normal / fearful from annualized vol (%)."""
    v = float(np.mean(ann_vol_pct))
    if v < 15:
        return "CALM", v
    if v < 30:
        return "NORMAL", v
    return "FEARFUL", v


def analyze(returns, horizon=30):
    res = fit_garch(returns)
    fc = forecast_vol(res, horizon)
    regime, level = classify_regime(fc)
    return {"res": res, "forecast": fc, "regime": regime,
            "regime_level": level,
            "hist_vol": historical_vol(returns) * 100.0,
            "conditional_vol": np.asarray(res.conditional_volatility) * np.sqrt(252)}

# garch.py
# --------
# GARCH(1,1) volatility model.
#
# Markets have vol clustering — big moves follow big moves, calm follows calm.
# Plain historical vol misses this completely. GARCH captures it:
#
#   sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
#
# omega  = baseline variance floor
# alpha  = how much yesterday's shock feeds into today's variance
# beta   = how persistent variance is (beta near 1 = long memory)
# alpha + beta must be < 1 for the model to be stationary
#
# Long-run variance = omega / (1 - alpha - beta)
# Forecast just iterates the equation forward from the last fitted variance.
#
# We use the arch library for MLE fitting. If it's not installed or fails,
# there's an EWMA fallback that gives similar intuition without full MLE.
#
# Regime classification is rough but useful for display:
#   <15% annualised = calm, 15-25% = normal, >25% = stressed

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal


@dataclass
class GARCHResult:
    omega:           float
    alpha:           float
    beta:            float
    persistence:     float       # alpha + beta
    long_run_vol:    float
    current_vol:     float
    regime:          str
    forecast_vol:    np.ndarray  # 30-day forward, annualised
    hist_vol_30d:    float
    hist_vol_252d:   float
    lower_band:      np.ndarray
    upper_band:      np.ndarray
    conditional_vol: np.ndarray  # in-sample


class GARCHModel:
    """
    Fit GARCH(1,1) on daily log-returns and forecast 30 days forward.
    Input should be a pd.Series of daily log-returns.
    """

    def __init__(self, returns: pd.Series):
        if isinstance(returns, pd.DataFrame):
            returns = returns.squeeze()
        self.returns = returns.dropna() * 100  # scale to % for numerical stability

    def fit_and_forecast(self, horizon: int = 30) -> GARCHResult:
        try:
            return self._fit_arch(horizon)
        except Exception:
            return self._fit_fallback(horizon)

    def _fit_arch(self, horizon: int) -> GARCHResult:
        from arch import arch_model

        am  = arch_model(self.returns, vol="Garch", p=1, q=1, dist="normal", mean="Constant")
        res = am.fit(disp="off", show_warning=False)

        params = res.params
        omega  = float(params.get("omega",    params.iloc[1]))
        alpha  = float(params.get("alpha[1]", params.iloc[2]))
        beta   = float(params.get("beta[1]",  params.iloc[3]))

        cond_var_daily = res.conditional_volatility ** 2
        cond_vol_ann   = (cond_var_daily ** 0.5) / 100 * np.sqrt(252)

        fc           = res.forecast(horizon=horizon, reindex=False)
        fc_var_daily = fc.variance.values[-1]
        fc_vol_ann   = np.sqrt(fc_var_daily) / 100 * np.sqrt(252)

        current_vol  = float(cond_var_daily.iloc[-1] ** 0.5) / 100 * np.sqrt(252)
        long_run_vol = (omega / max(1 - alpha - beta, 1e-6)) ** 0.5 / 100 * np.sqrt(252)

        vol_of_vol  = float(np.std(cond_vol_ann)) * 0.5
        lower_band  = np.maximum(fc_vol_ann - vol_of_vol, 0.01)
        upper_band  = fc_vol_ann + vol_of_vol

        return self._build_result(omega, alpha, beta, current_vol, long_run_vol,
                                  fc_vol_ann, lower_band, upper_band, cond_vol_ann.values)

    def _fit_fallback(self, horizon: int) -> GARCHResult:
        # EWMA with lambda=0.94 (RiskMetrics standard)
        # Not full MLE but captures the same variance persistence idea
        r   = self.returns.values / 100
        lam = 0.94

        cond_var    = np.zeros(len(r))
        cond_var[0] = np.var(r)
        for t in range(1, len(r)):
            cond_var[t] = lam * cond_var[t-1] + (1 - lam) * r[t-1]**2

        alpha = 1 - lam
        beta  = lam
        omega = np.var(r) * (1 - alpha - beta) * 252

        # forecast: mean-revert to long-run variance
        fc_var = np.zeros(horizon)
        v      = cond_var[-1]
        for h in range(horizon):
            v = omega/252 + alpha * r[-1]**2 + beta * v
            fc_var[h] = v

        fc_vol_ann   = np.sqrt(fc_var) * np.sqrt(252)
        cond_vol_ann = np.sqrt(cond_var) * np.sqrt(252)
        current_vol  = float(cond_vol_ann[-1])
        long_run_vol = float(np.std(r) * np.sqrt(252))
        vol_of_vol   = float(np.std(cond_vol_ann)) * 0.5

        return self._build_result(omega, alpha, beta, current_vol, long_run_vol,
                                  fc_vol_ann,
                                  np.maximum(fc_vol_ann - vol_of_vol, 0.01),
                                  fc_vol_ann + vol_of_vol,
                                  cond_vol_ann)

    def _build_result(self, omega, alpha, beta, current_vol, long_run_vol,
                      fc_vol_ann, lower_band, upper_band, cond_vol_ann) -> GARCHResult:
        raw      = self.returns.values / 100
        hist_30  = float(np.std(raw[-30:])  * np.sqrt(252)) if len(raw) >= 30  else current_vol
        hist_252 = float(np.std(raw[-252:]) * np.sqrt(252)) if len(raw) >= 252 else current_vol

        return GARCHResult(
            omega=omega, alpha=alpha, beta=beta,
            persistence    = alpha + beta,
            long_run_vol   = long_run_vol,
            current_vol    = current_vol,
            regime         = self._regime(current_vol),
            forecast_vol   = fc_vol_ann,
            hist_vol_30d   = hist_30,
            hist_vol_252d  = hist_252,
            lower_band     = lower_band,
            upper_band     = upper_band,
            conditional_vol= cond_vol_ann,
        )

    @staticmethod
    def _regime(vol: float) -> Literal["calm", "normal", "fearful"]:
        if vol < 0.15:   return "calm"
        elif vol < 0.25: return "normal"
        else:            return "fearful"

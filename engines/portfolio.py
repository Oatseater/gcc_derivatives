# portfolio.py
# ------------
# Markowitz efficient frontier via random portfolio sampling.
#
# The idea from Markowitz (1952): for any level of risk, there's an optimal
# portfolio that maximises return. The set of these portfolios is the efficient
# frontier. Anything below it is suboptimal — same risk, less return.
#
# We sample 500 random weight vectors (Dirichlet distribution = uniform over
# the simplex) rather than solving the full QP. 500 is enough to get a dense,
# visually clear frontier and it's way more robust than optimisation when
# the covariance matrix is near-singular (which it often is with GCC data).
#
# Two special portfolios:
#   Max Sharpe  = best risk-adjusted return: (E[r] - rf) / sigma
#   Min Variance = lowest possible vol, ignoring return
#
# Correlation matrix tells you diversification benefit.
# Low correlation between assets = the frontier bows further left.

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PortfolioPoint:
    weights:    np.ndarray
    tickers:    list[str]
    exp_return: float
    volatility: float
    sharpe:     float


@dataclass
class FrontierResult:
    tickers:      list[str]
    returns_arr:  np.ndarray
    vols_arr:     np.ndarray
    sharpes_arr:  np.ndarray
    weights_arr:  np.ndarray
    max_sharpe:   PortfolioPoint
    min_vol:      PortfolioPoint
    cov_matrix:   pd.DataFrame
    corr_matrix:  pd.DataFrame
    mean_returns: pd.Series


class PortfolioOptimiser:
    """
    price_dict: output of fetcher.fetch_multiple — dict of {ticker: ohlcv_df}
    risk_free:  annual risk-free rate for Sharpe calculation
    n_portfolios: how many random weights to try (500 is plenty)
    """

    def __init__(self, price_dict, risk_free=0.05, n_portfolios=500, seed=42):
        self.risk_free    = risk_free
        self.n_portfolios = n_portfolios
        self.rng          = np.random.default_rng(seed)

        closes  = {t: df["Close"].rename(t) for t, df in price_dict.items() if not df.empty}
        prices  = pd.concat(closes, axis=1).dropna()
        self.returns  = prices.pct_change().dropna()
        self.tickers  = list(self.returns.columns)
        self.n_assets = len(self.tickers)

    def compute(self) -> FrontierResult:
        mu   = self.returns.mean() * 252
        cov  = self.returns.cov()  * 252
        corr = self.returns.corr()

        ret_arr = np.zeros(self.n_portfolios)
        vol_arr = np.zeros(self.n_portfolios)
        sh_arr  = np.zeros(self.n_portfolios)
        w_arr   = np.zeros((self.n_portfolios, self.n_assets))

        for i in range(self.n_portfolios):
            w              = self._random_weights()
            r, v, s        = self._stats(w, mu.values, cov.values)
            ret_arr[i]     = r
            vol_arr[i]     = v
            sh_arr[i]      = s
            w_arr[i]       = w

        ms = int(np.argmax(sh_arr))
        mv = int(np.argmin(vol_arr))

        return FrontierResult(
            tickers      = self.tickers,
            returns_arr  = ret_arr,
            vols_arr     = vol_arr,
            sharpes_arr  = sh_arr,
            weights_arr  = w_arr,
            max_sharpe   = self._point(w_arr[ms], mu, ret_arr[ms], vol_arr[ms], sh_arr[ms]),
            min_vol      = self._point(w_arr[mv], mu, ret_arr[mv], vol_arr[mv], sh_arr[mv]),
            cov_matrix   = cov,
            corr_matrix  = corr,
            mean_returns = mu,
        )

    def _random_weights(self) -> np.ndarray:
        # exponential trick: normalised exponentials = uniform on simplex
        w = self.rng.exponential(1.0, self.n_assets)
        return w / w.sum()

    def _stats(self, w, mu, cov):
        ret    = float(w @ mu)
        vol    = float(np.sqrt(max(w @ cov @ w, 1e-12)))
        sharpe = (ret - self.risk_free) / vol
        return ret, vol, sharpe

    def _point(self, w, mu, ret, vol, sharpe):
        return PortfolioPoint(weights=w, tickers=self.tickers,
                              exp_return=ret, volatility=vol, sharpe=sharpe)

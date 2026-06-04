# monte_carlo.py
# --------------
# Monte Carlo risk engine using Geometric Brownian Motion.
#
# The idea: instead of a closed-form solution, simulate thousands of random
# price paths and look at the distribution of outcomes. GBM says:
#
#   S(t+dt) = S(t) * exp[(mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z]
#
# where Z ~ N(0,1). We use the exact log-normal formula rather than Euler
# discretisation so there's no drift in the simulation.
#
# With 10k paths you get stable VaR estimates. More paths = smoother distribution
# but slower. 1k is fine for quick exploration, 10k for anything serious.
#
# VaR vs CVaR:
#   VaR(95%) = the loss you won't exceed 95% of the time
#   CVaR(95%) = your expected loss when you DO exceed it (the tail average)
#   CVaR is the better risk measure — VaR ignores what happens in the tail

import numpy as np
from dataclasses import dataclass


@dataclass
class MCResult:
    paths:          np.ndarray
    final_prices:   np.ndarray
    returns:        np.ndarray
    var_95:         float
    var_99:         float
    cvar_95:        float
    cvar_99:        float
    mean_return:    float
    std_return:     float
    prob_loss:      float


@dataclass
class StressResult:
    scenario:        str
    shock:           float
    stressed_price:  float
    stressed_return: float


class MonteCarlo:
    """
    GBM simulator. Default 10k paths, 252 steps (one per trading day).
    
    S0    = starting price
    mu    = expected annual return (use risk-free rate for risk-neutral pricing)
    sigma = annual vol
    T     = horizon in years
    N     = number of paths
    """

    STRESS_SCENARIOS = {
        "2008 Financial Crisis": -0.40,
        "COVID-19 Crash":        -0.30,
        "GCC Gulf Crisis":       -0.20,
        "Dot-com Bust":          -0.35,
        "Black Monday 1987":     -0.22,
    }

    def __init__(self, S0, mu, sigma, T=1.0, N=10_000, steps=252, seed=42):
        self.S0    = S0
        self.mu    = mu
        self.sigma = sigma
        self.T     = T
        self.N     = N
        self.steps = steps
        self.seed  = seed

    def simulate(self) -> MCResult:
        rng = np.random.default_rng(self.seed)
        dt  = self.T / self.steps

        Z           = rng.standard_normal((self.N, self.steps))
        log_returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
        cum_log     = np.concatenate([np.zeros((self.N, 1)), np.cumsum(log_returns, axis=1)], axis=1)
        paths       = self.S0 * np.exp(cum_log)
        final       = paths[:, -1]
        rets        = (final - self.S0) / self.S0

        var_95, cvar_95 = self._var_cvar(rets, 0.95)
        var_99, cvar_99 = self._var_cvar(rets, 0.99)

        return MCResult(
            paths        = paths,
            final_prices = final,
            returns      = rets,
            var_95       = var_95,
            var_99       = var_99,
            cvar_95      = cvar_95,
            cvar_99      = cvar_99,
            mean_return  = float(np.mean(rets)),
            std_return   = float(np.std(rets)),
            prob_loss    = float(np.mean(rets < 0)),
        )

    def stress_test(self) -> list[StressResult]:
        # apply historical crash magnitudes as instant shocks
        return [
            StressResult(s, shock, self.S0 * (1 + shock), shock)
            for s, shock in self.STRESS_SCENARIOS.items()
        ]

    @staticmethod
    def _var_cvar(returns: np.ndarray, confidence: float) -> tuple[float, float]:
        alpha     = 1.0 - confidence
        var       = float(-np.percentile(returns, alpha * 100))
        tail_mask = returns < -var
        cvar      = float(-np.mean(returns[tail_mask])) if tail_mask.any() else var
        return var, cvar

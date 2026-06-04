# black_scholes.py
# ----------------
# Pricing European options from scratch. No libraries — just numpy and scipy.norm.
#
# Quick primer if you need it:
#   BS gives the "fair" price of a call/put assuming log-normal returns, constant vol,
#   and a frictionless market. In practice nobody believes those assumptions, but it's
#   still the market standard for quoting options (via implied vol).
#
#   d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
#   d2 = d1 - σ√T
#   C  = S·N(d1) - K·e^(-rT)·N(d2)
#   P  = K·e^(-rT)·N(-d2) - S·N(-d1)   (or just C - S + Ke^-rT by parity)
#
# Greeks are just partial derivatives of the price formula. Delta hedging means
# keeping your net delta near zero so small moves don't kill you.
#
# IV solver: given a market price, find σ that makes BS(σ) = market_price.
# Newton-Raphson works well here since vega (dPrice/dσ) is always positive.

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


@dataclass
class OptionResult:
    option_type: str
    price:  float
    delta:  float
    gamma:  float
    theta:  float   # per calendar day
    vega:   float   # per 1% vol move
    rho:    float   # per 1% rate move
    d1:     float
    d2:     float


class BlackScholes:
    """
    Standard BS pricer. Pass in S, K, T, r, sigma and call compute_all().

    S     = current spot price
    K     = strike
    T     = time to expiry in years (30 days = 30/365)
    r     = risk-free rate, continuously compounded (0.05 = 5%)
    sigma = annualised vol (0.20 = 20%)
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        self.S     = np.float64(S)
        self.K     = np.float64(K)
        self.T     = np.float64(T)
        self.r     = np.float64(r)
        self.sigma = np.float64(sigma)
        self._d1, self._d2 = self._compute_d()

    def _compute_d(self):
        eps = 1e-10
        T_  = max(self.T, eps)
        sig = max(self.sigma, eps)
        d1  = (np.log(self.S / self.K) + (self.r + 0.5 * sig**2) * T_) / (sig * np.sqrt(T_))
        d2  = d1 - sig * np.sqrt(T_)
        return d1, d2

    def call_price(self) -> float:
        return (self.S * norm.cdf(self._d1)
                - self.K * np.exp(-self.r * self.T) * norm.cdf(self._d2))

    def put_price(self) -> float:
        # put-call parity: P = C - S + Ke^(-rT)
        return self.call_price() - self.S + self.K * np.exp(-self.r * self.T)

    def price(self, option_type: Literal["call", "put"] = "call") -> float:
        return self.call_price() if option_type == "call" else self.put_price()

    def delta(self, option_type: Literal["call", "put"] = "call") -> float:
        # how much the option price moves per $1 move in spot
        # call: (0,1), put: (-1,0)
        if option_type == "call":
            return norm.cdf(self._d1)
        return norm.cdf(self._d1) - 1.0

    def gamma(self) -> float:
        # rate of change of delta — same for calls and puts
        # spikes near expiry for ATM options (gamma risk)
        return norm.pdf(self._d1) / (self.S * self.sigma * np.sqrt(max(self.T, 1e-10)))

    def theta(self, option_type: Literal["call", "put"] = "call") -> float:
        # daily time decay — almost always negative
        T_  = max(self.T, 1e-10)
        sig = max(self.sigma, 1e-10)
        common = -(self.S * norm.pdf(self._d1) * sig) / (2 * np.sqrt(T_))
        if option_type == "call":
            th = common - self.r * self.K * np.exp(-self.r * T_) * norm.cdf(self._d2)
        else:
            th = common + self.r * self.K * np.exp(-self.r * T_) * norm.cdf(-self._d2)
        return th / 365.0

    def vega(self) -> float:
        # sensitivity to a 1% change in vol (divided by 100)
        # same for calls and puts
        return self.S * norm.pdf(self._d1) * np.sqrt(max(self.T, 1e-10)) / 100.0

    def rho(self, option_type: Literal["call", "put"] = "call") -> float:
        # sensitivity to a 1% change in interest rate
        T_ = max(self.T, 1e-10)
        if option_type == "call":
            return self.K * T_ * np.exp(-self.r * T_) * norm.cdf(self._d2) / 100.0
        return -self.K * T_ * np.exp(-self.r * T_) * norm.cdf(-self._d2) / 100.0

    def compute_all(self, option_type: Literal["call", "put"] = "call") -> OptionResult:
        return OptionResult(
            option_type = option_type,
            price  = self.price(option_type),
            delta  = self.delta(option_type),
            gamma  = self.gamma(),
            theta  = self.theta(option_type),
            vega   = self.vega(),
            rho    = self.rho(option_type),
            d1     = self._d1,
            d2     = self._d2,
        )


def implied_volatility(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: Literal["call", "put"] = "call",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    Newton-Raphson IV solver.
    
    Start at σ=0.20, iterate: σ -= (BS(σ) - market_price) / vega
    Converges in <10 steps for most liquid options.
    Returns nan if it can't converge (deep ITM/OTM or bad input).
    """
    T_ = max(T, 1e-10)
    intrinsic = max(S - K * np.exp(-r * T_), 0) if option_type == "call" else max(K * np.exp(-r * T_) - S, 0)
    if market_price < intrinsic - 1e-4:
        return float("nan")

    sigma = 0.20
    for _ in range(max_iter):
        bs      = BlackScholes(S, K, T, r, sigma)
        price_  = bs.price(option_type)
        vega_   = bs.vega() * 100
        if abs(vega_) < 1e-10:
            break
        diff    = price_ - market_price
        if abs(diff) < tol:
            return sigma
        sigma  -= diff / vega_
        sigma   = max(sigma, 1e-6)

    return sigma if abs(BlackScholes(S, K, T, r, sigma).price(option_type) - market_price) < 0.01 else float("nan")


def iv_surface(
    S: float, r: float,
    strikes_pct: np.ndarray | None = None,
    expiries: np.ndarray | None = None,
    base_vol: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic IV surface with a volatility smile baked in.
    
    Real surfaces have a skew (OTM puts are expensive — everyone wants downside protection)
    and a term structure (longer dated = higher vol generally).
    We model it as: IV = base_vol + skew*moneyness + smile*moneyness^2 + term_adj
    
    Returns (K_grid, T_grid, IV_grid) for 3D plotting.
    """
    if strikes_pct is None:
        strikes_pct = np.linspace(0.70, 1.30, 20)
    if expiries is None:
        expiries = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])

    K_arr   = strikes_pct * S
    K_grid, T_grid = np.meshgrid(K_arr, expiries)

    moneyness = np.log(K_grid / S) / np.sqrt(T_grid)
    skew      = -0.05   # puts more expensive than calls
    smile     = 0.04    # smile curvature
    term_adj  = 0.03 * (1 - np.exp(-T_grid))

    IV_grid = base_vol + skew * moneyness + smile * moneyness**2 + term_adj
    IV_grid = np.clip(IV_grid, 0.05, 1.50)

    return K_grid, T_grid, IV_grid

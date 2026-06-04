"""
utils/charts.py
---------------
Plotly Chart Factory

All charts use a consistent dark-terminal aesthetic:
    Background : transparent (overlays onto Streamlit dark theme)
    Grid       : subtle dark lines (#1e2130)
    Accent     : blue  #4a9eff  |  red #ff4d4f  |  green #52c41a  |  amber #faad14

Each function returns a go.Figure ready to pass to st.plotly_chart().
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from engines.black_scholes import iv_surface, BlackScholes
from engines.monte_carlo   import MCResult
from engines.garch         import GARCHResult
from engines.portfolio     import FrontierResult


# ── Shared style helpers ──────────────────────────────────────────────────────

BLUE   = "#4a9eff"
RED    = "#ff4d4f"
GREEN  = "#52c41a"
AMBER  = "#faad14"
PURPLE = "#9b59b6"
BG     = "rgba(0,0,0,0)"
GRID   = "#1e2130"
TEXT   = "#c9d1d9"
FONT   = "IBM Plex Mono, monospace"


def _base_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor = BG,
        plot_bgcolor  = BG,
        font          = dict(family=FONT, color=TEXT, size=12),
        margin        = dict(l=50, r=30, t=50, b=50),
        xaxis         = dict(gridcolor=GRID, zerolinecolor=GRID, showgrid=True),
        yaxis         = dict(gridcolor=GRID, zerolinecolor=GRID, showgrid=True),
        **kwargs,
    )


def _apply_base(fig: go.Figure, title: str = "", **extra) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(color=TEXT, size=14)), **_base_layout(**extra))
    return fig


# ── 1. IV Surface (3-D Mesh) ──────────────────────────────────────────────────

def iv_surface_chart(S: float, sigma: float) -> go.Figure:
    """
    3-D implied volatility surface showing the volatility smile/skew.

    The x-axis is strike (% of spot), y-axis is time to expiry, z-axis is IV.
    The characteristic 'smile' arises from demand for OTM puts (downside protection).
    """
    K_grid, T_grid, IV_grid = iv_surface(S=S, base_vol=sigma)

    fig = go.Figure(data=[
        go.Surface(
            x = K_grid / S * 100,   # moneyness %
            y = T_grid * 12,        # months
            z = IV_grid * 100,      # IV as %
            colorscale  = "Blues",
            opacity     = 0.88,
            showscale   = True,
            colorbar    = dict(title="IV %", tickfont=dict(color=TEXT)),
            lighting    = dict(ambient=0.6, diffuse=0.8),
        )
    ])

    fig.update_layout(
        title  = "Implied Volatility Surface",
        scene  = dict(
            xaxis = dict(title="Strike (% spot)", gridcolor=GRID, backgroundcolor=BG, color=TEXT),
            yaxis = dict(title="Expiry (months)", gridcolor=GRID, backgroundcolor=BG, color=TEXT),
            zaxis = dict(title="IV (%)",          gridcolor=GRID, backgroundcolor=BG, color=TEXT),
            bgcolor = BG,
        ),
        paper_bgcolor = BG,
        font = dict(family=FONT, color=TEXT),
        height = 500,
    )
    return fig


# ── 2. Monte Carlo Paths ──────────────────────────────────────────────────────

def mc_paths_chart(result: MCResult, S0: float, n_plot: int = 200) -> go.Figure:
    """
    Fan chart of Monte Carlo price paths, coloured by final terminal value.

    Plotting all 10 000 paths would freeze the browser; we subsample 200.
    Paths ending above S0 are in blue shades; below in red — giving immediate
    visual intuition of the left-tail skew.
    """
    paths = result.paths
    steps = paths.shape[1]
    x_axis = np.linspace(0, 1, steps)

    rng = np.random.default_rng(99)
    idx = rng.choice(len(paths), size=min(n_plot, len(paths)), replace=False)
    sub_paths = paths[idx]
    finals    = sub_paths[:, -1]

    # Normalise final price for colour mapping
    f_min, f_max = finals.min(), finals.max()
    norm = (finals - f_min) / max(f_max - f_min, 1e-6)

    fig = go.Figure()
    for i, (path, nv) in enumerate(zip(sub_paths, norm)):
        colour = f"rgba({int(74 + (255-74)*nv)},{int(158 + (77-158)*nv)},{int(255 + (77-255)*nv)},0.25)"
        fig.add_trace(go.Scatter(
            x=x_axis, y=path,
            mode="lines",
            line=dict(color=colour, width=0.6),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Mean path
    mean_path = paths.mean(axis=0)
    fig.add_trace(go.Scatter(
        x=x_axis, y=mean_path,
        mode="lines",
        line=dict(color=AMBER, width=2.5, dash="dot"),
        name="Mean Path",
    ))

    # S0 reference line
    fig.add_hline(y=S0, line=dict(color=TEXT, width=1, dash="dash"), opacity=0.4)

    _apply_base(fig, title=f"Monte Carlo — {len(paths):,} GBM Paths (showing {len(sub_paths)})")
    fig.update_layout(
        xaxis_title="Time (fraction of horizon)",
        yaxis_title="Price",
        height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
    )
    return fig


# ── 3. Return Distribution + VaR markers ─────────────────────────────────────

def mc_distribution_chart(result: MCResult) -> go.Figure:
    """
    Histogram of simulated terminal returns with VaR/CVaR overlays.

    The shaded tail region beyond 95% VaR visually represents the loss
    scenarios that define the Expected Shortfall (CVaR).
    """
    rets = result.returns * 100  # convert to %

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets,
        nbinsx=80,
        marker_color=BLUE,
        opacity=0.7,
        name="Simulated Returns",
    ))

    for var, label, color in [
        (result.var_95, "95% VaR", AMBER),
        (result.var_99, "99% VaR", RED),
    ]:
        fig.add_vline(
            x=-var*100,
            line=dict(color=color, width=2, dash="dash"),
            annotation=dict(text=f"  {label}: {var*100:.1f}%", font=dict(color=color, size=11)),
        )

    _apply_base(fig, title="Return Distribution — Terminal Returns")
    fig.update_layout(
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        height=350,
        bargap=0.05,
        legend=dict(bgcolor=BG, font=dict(color=TEXT)),
    )
    return fig


# ── 4. Greeks Heatmap ─────────────────────────────────────────────────────────

def greeks_heatmap(S: float, K: float, r: float, sigma: float, option_type: str) -> go.Figure:
    """
    Heatmap of Delta vs (Spot Price × Time to Expiry).

    Shows how delta evolves as the underlying moves and time decays —
    essential for understanding the dynamic hedging requirements of a position.
    """
    spots   = np.linspace(S * 0.70, S * 1.30, 30)
    expiries= np.linspace(0.02, 1.0, 20)

    delta_matrix = np.zeros((len(expiries), len(spots)))
    for i, t in enumerate(expiries):
        for j, s in enumerate(spots):
            bs = BlackScholes(s, K, t, r, sigma)
            delta_matrix[i, j] = bs.delta(option_type)

    fig = go.Figure(data=go.Heatmap(
        z          = delta_matrix,
        x          = np.round(spots, 2),
        y          = np.round(expiries * 12, 1),
        colorscale = [[0, RED], [0.5, "#1e2130"], [1, BLUE]],
        colorbar   = dict(title="Delta", tickfont=dict(color=TEXT)),
        hoverongaps= False,
        hovertemplate="Spot: %{x}<br>Expiry (mo): %{y}<br>Delta: %{z:.3f}<extra></extra>",
    ))

    _apply_base(fig, title=f"Delta Heatmap — {option_type.title()} Option")
    fig.update_layout(
        xaxis_title="Spot Price",
        yaxis_title="Expiry (months)",
        height=380,
    )
    return fig


# ── 5. GARCH Volatility Forecast ──────────────────────────────────────────────

def garch_forecast_chart(result: GARCHResult, historical_dates: pd.DatetimeIndex | None = None) -> go.Figure:
    """
    Two-panel chart: left = historical conditional vol, right = 30-day forecast.

    The shaded band represents ±1σ uncertainty around the point forecast.
    The horizontal dashed lines mark the calm/fearful regime thresholds.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Historical Conditional Volatility", "30-Day Forecast"],
        column_widths=[0.55, 0.45],
    )

    # ── Left: historical ──────────────────────────────────────────────────────
    n_hist = len(result.conditional_vol)
    if historical_dates is not None and len(historical_dates) >= n_hist:
        x_hist = historical_dates[-n_hist:]
    else:
        x_hist = np.arange(n_hist)

    fig.add_trace(go.Scatter(
        x=x_hist, y=result.conditional_vol * 100,
        mode="lines", line=dict(color=BLUE, width=1.5),
        name="Conditional Vol",
    ), row=1, col=1)

    for thresh, label, color in [
        (15, "Calm threshold", GREEN),
        (25, "Fearful threshold", RED),
    ]:
        fig.add_hline(
            y=thresh, row=1, col=1,
            line=dict(color=color, width=1, dash="dot"),
        )

    # ── Right: forecast ───────────────────────────────────────────────────────
    x_fc = np.arange(1, len(result.forecast_vol) + 1)
    fc   = result.forecast_vol * 100
    lo   = result.lower_band   * 100
    hi   = result.upper_band   * 100

    # Confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fc, x_fc[::-1]]),
        y=np.concatenate([hi, lo[::-1]]),
        fill="toself",
        fillcolor=f"rgba(74,158,255,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        name="±1σ Band",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=x_fc, y=fc,
        mode="lines+markers",
        line=dict(color=AMBER, width=2.5),
        marker=dict(size=4),
        name="Vol Forecast",
    ), row=1, col=2)

    fig.add_hline(
        y=result.long_run_vol * 100, row=1, col=2,
        line=dict(color=PURPLE, width=1.5, dash="dash"),
        annotation=dict(text="Long-run vol", font=dict(color=PURPLE, size=10)),
    )

    _apply_base(fig, title="GARCH(1,1) Volatility Analysis")
    fig.update_layout(
        height=400,
        yaxis_title="Annualised Vol (%)",
        yaxis2_title="Annualised Vol (%)",
        xaxis2_title="Days ahead",
        legend=dict(bgcolor=BG, font=dict(color=TEXT)),
    )
    fig.update_annotations(font=dict(color=TEXT, size=12))
    return fig


# ── 6. Efficient Frontier ─────────────────────────────────────────────────────

def efficient_frontier_chart(result: FrontierResult) -> go.Figure:
    """
    Scatter plot of 500 random portfolios on the Risk-Return plane,
    coloured by Sharpe ratio.

    The concave boundary of the scatter IS the efficient frontier — any
    portfolio below-left of this boundary is dominated (same risk, lower return).
    The gold star marks the maximum Sharpe portfolio; the cyan diamond marks
    the minimum variance portfolio.
    """
    fig = go.Figure()

    # All random portfolios
    fig.add_trace(go.Scatter(
        x    = result.vols_arr   * 100,
        y    = result.returns_arr * 100,
        mode = "markers",
        marker = dict(
            color     = result.sharpes_arr,
            colorscale= "Blues",
            size      = 5,
            opacity   = 0.7,
            colorbar  = dict(title="Sharpe", tickfont=dict(color=TEXT)),
        ),
        name      = "Portfolios",
        hovertemplate = "Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))

    # Max Sharpe star
    ms = result.max_sharpe
    fig.add_trace(go.Scatter(
        x=[ms.volatility*100], y=[ms.exp_return*100],
        mode="markers+text",
        marker=dict(symbol="star", size=18, color=AMBER),
        text=["Max SR"], textposition="top right",
        textfont=dict(color=AMBER, size=11),
        name="Max Sharpe",
    ))

    # Min Vol diamond
    mv = result.min_vol
    fig.add_trace(go.Scatter(
        x=[mv.volatility*100], y=[mv.exp_return*100],
        mode="markers+text",
        marker=dict(symbol="diamond", size=14, color=GREEN),
        text=["Min Vol"], textposition="top right",
        textfont=dict(color=GREEN, size=11),
        name="Min Vol",
    ))

    _apply_base(fig, title="Efficient Frontier — Mean-Variance Optimisation")
    fig.update_layout(
        xaxis_title="Portfolio Volatility (%)",
        yaxis_title="Expected Return (%)",
        height=460,
        legend=dict(bgcolor=BG, font=dict(color=TEXT)),
    )
    return fig


# ── 7. Correlation Matrix ─────────────────────────────────────────────────────

def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """
    Heatmap of pairwise asset correlations.
    Low correlations = diversification benefit.
    """
    fig = go.Figure(data=go.Heatmap(
        z          = corr.values,
        x          = corr.columns.tolist(),
        y          = corr.index.tolist(),
        colorscale = [[0, RED], [0.5, "#0d0f14"], [1, BLUE]],
        zmin=-1, zmax=1,
        colorbar   = dict(title="ρ", tickfont=dict(color=TEXT)),
        text       = np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont   = dict(size=11, color=TEXT),
        hoverongaps= False,
    ))

    _apply_base(fig, title="Asset Correlation Matrix")
    fig.update_layout(height=350)
    return fig

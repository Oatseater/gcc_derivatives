"""Plotly chart builders — Porsche-matte terminal aesthetic."""
import numpy as np
import plotly.graph_objects as go

BG = "#080808"
GRID = "#111111"
ZERO = "#222222"
GOLD = "#C9A84C"
WHITE = "#E8E8E8"
RED = "#E8192C"
FONT = "DM Mono, monospace"

VOL_SCALE = [[0.0, "#080808"], [0.5, "#0a1628"], [1.0, "#C9A84C"]]


def _base_layout(fig, height=420):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color=WHITE, size=11),
        margin=dict(l=50, r=30, t=40, b=40), height=height,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=ZERO, linecolor=GRID,
                     tickfont=dict(family=FONT, color="#888"))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=ZERO, linecolor=GRID,
                     tickfont=dict(family=FONT, color="#888"))
    return fig


def vol_surface(strikes, maturities, ivs):
    """3D implied vol surface. ivs shape (len(maturities), len(strikes))."""
    fig = go.Figure(go.Surface(
        x=strikes, y=maturities, z=ivs, colorscale=VOL_SCALE,
        showscale=False, opacity=0.95,
        contours={"z": {"show": True, "color": GOLD, "width": 1}},
    ))
    fig.update_layout(
        paper_bgcolor=BG, font=dict(family=FONT, color=WHITE, size=10),
        margin=dict(l=0, r=0, t=20, b=0), height=460,
        scene=dict(
            xaxis=dict(title="Strike", backgroundcolor=BG, gridcolor=GRID,
                       color="#888", showbackground=False),
            yaxis=dict(title="Maturity", backgroundcolor=BG, gridcolor=GRID,
                       color="#888", showbackground=False),
            zaxis=dict(title="IV", backgroundcolor=BG, gridcolor=GRID,
                       color="#888", showbackground=False),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
    )
    return fig


def monte_carlo_paths(paths, show=200):
    """Plot subset of paths colored by final value; gold mean path."""
    n = paths.shape[0]
    idx = np.random.choice(n, min(show, n), replace=False)
    fig = go.Figure()
    finals = paths[idx, -1]
    base = finals[len(finals) // 2] if len(finals) else paths[0, 0]
    x = np.arange(paths.shape[1])
    for i in idx:
        col = GOLD if paths[i, -1] >= paths[i, 0] else RED
        fig.add_trace(go.Scatter(x=x, y=paths[i], mode="lines",
                                 line=dict(color=col, width=0.5),
                                 opacity=0.15, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=paths.mean(axis=0), mode="lines",
                             line=dict(color=GOLD, width=2),
                             opacity=1.0, name="Mean"))
    _base_layout(fig)
    fig.update_xaxes(title="Steps")
    fig.update_yaxes(title="Price")
    return fig


def greeks_heatmap(spots, times, values, label="Delta"):
    """Heatmap of a greek vs spot (x) and time (y)."""
    fig = go.Figure(go.Heatmap(
        x=spots, y=times, z=values, colorscale=VOL_SCALE, showscale=True,
        colorbar=dict(title=label, tickfont=dict(color="#888"), outlinewidth=0),
    ))
    _base_layout(fig)
    fig.update_xaxes(title="Spot")
    fig.update_yaxes(title="Time (yrs)")
    return fig


def efficient_frontier(fr, max_sharpe, min_var):
    """Scatter colored by Sharpe with the two optimal points marked."""
    fig = go.Figure(go.Scatter(
        x=fr["vol"], y=fr["ret"], mode="markers",
        marker=dict(size=5, color=fr["sharpe"], colorscale=VOL_SCALE,
                    showscale=True, opacity=0.7,
                    colorbar=dict(title="Sharpe", tickfont=dict(color="#888"),
                                  outlinewidth=0)),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=[max_sharpe["vol"]], y=[max_sharpe["ret"]],
                             mode="markers", marker=dict(size=13, color=GOLD,
                             symbol="star", line=dict(color=WHITE, width=1))))
    fig.add_trace(go.Scatter(x=[min_var["vol"]], y=[min_var["ret"]],
                             mode="markers", marker=dict(size=11, color=WHITE,
                             symbol="diamond")))
    _base_layout(fig)
    fig.update_xaxes(title="Volatility (ann.)")
    fig.update_yaxes(title="Return (ann.)")
    return fig


def garch_forecast(hist_vol, forecast, conf=1.96):
    """Conditional vol history + forecast with confidence bands."""
    fig = go.Figure()
    h = np.asarray(hist_vol)
    f = np.asarray(forecast)
    xh = np.arange(len(h))
    xf = np.arange(len(h), len(h) + len(f))

    fig.add_trace(go.Scatter(x=xh, y=h, mode="lines",
                             line=dict(color=WHITE, width=1), opacity=0.6))
    upper = f * (1 + 0.15)
    lower = f * (1 - 0.15)
    fig.add_trace(go.Scatter(x=np.concatenate([xf, xf[::-1]]),
                             y=np.concatenate([upper, lower[::-1]]),
                             fill="toself", fillcolor="rgba(201,168,76,0.12)",
                             line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xf, y=f, mode="lines",
                             line=dict(color=GOLD, width=2)))
    _base_layout(fig)
    fig.update_xaxes(title="Days")
    fig.update_yaxes(title="Annualized Vol (%)")
    return fig

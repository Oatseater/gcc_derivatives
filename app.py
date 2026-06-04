"""GCC DERIVATIVES TERMINAL — quant dashboard. Matte-black Porsche aesthetic."""
import numpy as np
import pandas as pd
import streamlit as st

from engines import black_scholes as bs
from engines import monte_carlo as mc
from engines import garch as gh
from engines import portfolio as pf
from data import fetcher
from utils import charts as ch

st.set_page_config(page_title="GCC Derivatives Terminal",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────── THEME ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@200;300;400&display=swap');

:root{
  --bg-primary:#080808; --bg-secondary:#0f0f0f; --bg-tertiary:#141414;
  --accent:#C9A84C; --accent-red:#E8192C; --text:#E8E8E8; --muted:#444;
  --border:#1a1a1a;
}
.stApp{background:#080808;color:#E8E8E8;font-family:'DM Sans',sans-serif;}
section[data-testid="stSidebar"]{background:#050505;border-right:1px solid #1a1a1a;}
section[data-testid="stSidebar"] *{font-family:'DM Mono',monospace;}
section[data-testid="stSidebar"] label{color:#444 !important;font-size:11px;
  letter-spacing:.12em;text-transform:uppercase;}

/* header */
.term-header{font-family:'DM Sans';font-weight:200;font-size:32px;
  letter-spacing:.2em;color:#E8E8E8;margin:0;}
.term-sub{font-family:'DM Mono';color:#333;font-size:12px;letter-spacing:.15em;
  margin:2px 0 6px 0;}
.gold-line{height:1px;width:60%;background:#C9A84C;margin:0 0 18px 0;}

/* metric cards */
.metric-card{background:#0f0f0f;border:1px solid #1a1a1a;border-left:2px solid #C9A84C;
  padding:14px 16px;margin-bottom:10px;border-radius:0;}
.metric-label{font-family:'DM Mono';font-size:10px;text-transform:uppercase;
  color:#444;letter-spacing:.15em;}
.metric-value{font-family:'DM Mono';font-size:28px;color:#E8E8E8;
  letter-spacing:.08em;margin-top:4px;}
.metric-red .metric-value{color:#E8192C;}
.metric-gold .metric-value{color:#C9A84C;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{background:transparent;gap:28px;border-bottom:1px solid #1a1a1a;}
.stTabs [data-baseweb="tab"]{background:transparent;font-family:'DM Mono';
  color:#333;letter-spacing:.1em;font-size:13px;padding:6px 0;}
.stTabs [aria-selected="true"]{color:#C9A84C !important;
  border-bottom:1px solid #C9A84C;}

/* tables */
table{font-family:'DM Mono' !important;}
.stDataFrame{border:1px solid #1a1a1a;}

/* regime pill */
.regime{display:inline-block;font-family:'DM Mono';font-size:13px;
  letter-spacing:.18em;padding:8px 18px;border:1px solid #1a1a1a;}
.regime-calm{color:#C9A84C;border-left:2px solid #C9A84C;}
.regime-normal{color:#E8E8E8;border-left:2px solid #E8E8E8;}
.regime-fearful{color:#E8192C;border-left:2px solid #E8192C;}

h1,h2,h3{font-family:'DM Sans';font-weight:300;letter-spacing:-.03em;color:#E8E8E8;}
.stSpinner > div{border-top-color:#C9A84C !important;}
</style>
""", unsafe_allow_html=True)


def card(label, value, kind=""):
    cls = {"red": "metric-red", "gold": "metric-gold"}.get(kind, "")
    st.markdown(f"""<div class="metric-card {cls}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div></div>""", unsafe_allow_html=True)


# ─────────────────────────── HEADER ───────────────────────────
st.markdown('<div class="term-header">GCC DERIVATIVES TERMINAL</div>',
            unsafe_allow_html=True)
st.markdown('<div class="term-sub">DIFC · QUANTITATIVE DESK · '
            'BLACK-SCHOLES · MONTE CARLO · GARCH · MPT</div>',
            unsafe_allow_html=True)
st.markdown('<div class="gold-line"></div>', unsafe_allow_html=True)

# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    st.markdown("**PARAMETERS**")
    ticker = st.selectbox("Underlying", list(fetcher.GCC_TICKERS.keys()), index=3)
    S = st.slider("Spot  S", 10.0, 500.0, 100.0, 1.0)
    K = st.slider("Strike  K", 10.0, 500.0, 105.0, 1.0)
    T = st.slider("Maturity  T (yrs)", 0.05, 3.0, 1.0, 0.05)
    r = st.slider("Rate  r", 0.0, 0.15, 0.04, 0.005)
    sigma = st.slider("Vol  σ", 0.05, 1.0, 0.25, 0.01)
    kind = st.radio("Type", ["call", "put"], horizontal=True)
    st.markdown("---")
    n_paths = st.select_slider("MC Paths", [1000, 5000, 10000, 25000], 10000)

tab1, tab2, tab3, tab4 = st.tabs(
    ["OPTIONS PRICER", "MONTE CARLO", "VOLATILITY", "PORTFOLIO"])

# ─────────────────────────── TAB 1: PRICER ───────────────────────────
with tab1:
    p = bs.price(S, K, T, r, sigma, kind)
    g = bs.greeks(S, K, T, r, sigma, kind)
    c1, c2, c3 = st.columns(3)
    with c1: card(f"{kind.upper()} PRICE", f"{p:,.4f}", "gold")
    with c2: card("MONEYNESS  S/K", f"{S/K:.3f}")
    with c3:
        iv = bs.implied_vol(p, S, K, T, r, kind)
        card("IMPLIED VOL", f"{iv*100:.2f}%")

    st.markdown("###### GREEKS")
    gc = st.columns(5)
    for col, (name, val) in zip(gc, g.items()):
        with col: card(name.upper(), f"{val:+.4f}")

    st.markdown("###### DELTA SURFACE  ·  spot × time")
    spots = np.linspace(S * 0.6, S * 1.4, 30)
    times = np.linspace(0.05, max(T, 0.5), 30)
    Z = np.array([[bs.greeks(s, K, t, r, sigma, kind)["Delta"]
                   for s in spots] for t in times])
    st.plotly_chart(ch.greeks_heatmap(spots, times, Z, "Delta"),
                    use_container_width=True)

    with st.expander("IMPLIED VOLATILITY SURFACE (3D)"):
        strikes = np.linspace(K * 0.7, K * 1.3, 20)
        mats = np.linspace(0.1, max(T, 1.0), 20)
        smile = lambda kk: sigma * (1 + 0.4 * ((kk - K) / K) ** 2)
        ivs = np.array([[smile(kk) * (1 + 0.05 * np.sqrt(m))
                         for kk in strikes] for m in mats])
        st.plotly_chart(ch.vol_surface(strikes, mats, ivs),
                        use_container_width=True)

# ─────────────────────────── TAB 2: MONTE CARLO ───────────────────────────
with tab2:
    with st.spinner("CALCULATING..."):
        res = mc.summary(S, r, sigma, T, N=n_paths, steps=252, seed=42)
    risk = res["risk"]
    c = st.columns(4)
    with c[0]: card("VaR 95%", f"{risk['VaR_95']:,.2f}", "red")
    with c[1]: card("VaR 99%", f"{risk['VaR_99']:,.2f}", "red")
    with c[2]: card("CVaR 95%", f"{risk['CVaR_95']:,.2f}", "red")
    with c[3]: card("CVaR 99%", f"{risk['CVaR_99']:,.2f}", "red")

    c = st.columns(2)
    with c[0]: card("TERMINAL MEAN", f"{res['terminal_mean']:,.2f}", "gold")
    with c[1]: card("TERMINAL STD", f"{res['terminal_std']:,.2f}")

    st.markdown(f"###### {n_paths:,} GBM PATHS  ·  200 SHOWN")
    st.plotly_chart(ch.monte_carlo_paths(res["paths"], 200),
                    use_container_width=True)

    st.markdown("###### STRESS SCENARIOS")
    sdf = pd.DataFrame(res["stress"]).T
    sdf["shock"] = (sdf["shock"] * 100).map(lambda x: f"{x:.0f}%")
    sdf.columns = ["Shock", "Shocked Price", "Loss"]
    st.dataframe(sdf.style.format({"Shocked Price": "{:,.2f}", "Loss": "{:,.2f}"}),
                 use_container_width=True)

# ─────────────────────────── TAB 3: VOLATILITY ───────────────────────────
with tab3:
    tk = fetcher.GCC_TICKERS[ticker]
    with st.spinner("CALCULATING..."):
        rets = fetcher.get_returns(tk, period="2y")
        if len(rets) < 100:
            np.random.seed(1)
            rets = pd.Series(np.random.normal(0, sigma / np.sqrt(252), 500))
            src = "synthetic fallback"
        else:
            src = tk
        ga = gh.analyze(rets.values, horizon=30)

    regime = ga["regime"]
    rcls = {"CALM": "regime-calm", "NORMAL": "regime-normal",
            "FEARFUL": "regime-fearful"}[regime]
    c = st.columns([1, 1, 2])
    with c[0]: card("HIST VOL (ann)", f"{ga['hist_vol']:.2f}%")
    with c[1]: card("FORECAST AVG", f"{np.mean(ga['forecast']):.2f}%", "gold")
    with c[2]:
        st.markdown(f"<div class='metric-label'>REGIME · {src}</div>"
                    f"<div class='regime {rcls}'>{regime} · "
                    f"{ga['regime_level']:.1f}%</div>", unsafe_allow_html=True)

    st.markdown("###### GARCH(1,1)  ·  30-DAY VOL FORECAST")
    st.plotly_chart(ch.garch_forecast(ga["conditional_vol"], ga["forecast"]),
                    use_container_width=True)

# ─────────────────────────── TAB 4: PORTFOLIO ───────────────────────────
with tab4:
    with st.spinner("CALCULATING..."):
        panel = fetcher.fetch_close_panel(fetcher.GCC_TICKERS, period="2y")
        if panel.shape[1] < 2 or len(panel) < 50:
            np.random.seed(2)
            idx = pd.date_range("2023-01-01", periods=500)
            panel = pd.DataFrame(
                100 * np.cumprod(1 + np.random.normal(
                    0.0004, 0.012, (500, 4)), axis=0),
                index=idx, columns=["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D"])
        po = pf.analyze(panel, rf=r, n_port=500)

    ms, mv = po["max_sharpe"], po["min_var"]
    c = st.columns(3)
    with c[0]: card("MAX SHARPE", f"{ms['sharpe']:.3f}", "gold")
    with c[1]: card("MAX-SHARPE RET", f"{ms['ret']*100:.2f}%")
    with c[2]: card("MIN-VAR VOL", f"{mv['vol']*100:.2f}%")

    st.markdown("###### EFFICIENT FRONTIER  ·  ★ max-Sharpe  ◆ min-var")
    st.plotly_chart(ch.efficient_frontier(po["frontier"], ms, mv),
                    use_container_width=True)

    st.markdown("###### OPTIMAL WEIGHTS")
    wdf = pd.DataFrame({"Max Sharpe": ms["weights"],
                        "Min Variance": mv["weights"]})
    st.dataframe(wdf.style.format("{:.2%}"), use_container_width=True)

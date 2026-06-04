# app.py
# GCC Derivatives Terminal
# Run: streamlit run app.py

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="GCC Derivatives Terminal",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- CSS: proper terminal, not AI dashboard -------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2family=Geist+Mono:wght@300;400;500;600&family=Geist:wght@300;400;500&display=swap');

/* -- Reset -- */
html,body,[class*="css"]{
  font-family:'Geist Mono',monospace!important;
  background:#08090c!important;
  color:#e2e8f0!important;
}
.main .block-container{
  padding:0!important;
  max-width:100%!important;
}

/* -- Hide Streamlit chrome -- */
#MainMenu,footer,header,[data-testid="stToolbar"]{display:none!important}
[data-testid="stDecoration"]{display:none!important}
section[data-testid="stSidebar"]{display:none!important}

/* -- Top bar -- */
.gcc-topbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:0 20px;height:36px;
  background:#0d0f14;border-bottom:1px solid #1d2130;
}
.gcc-logo{font-size:11px;font-weight:600;letter-spacing:.14em;color:#3b82f6;text-transform:uppercase}
.gcc-logo span{color:#4a5568}
.gcc-tickers{display:flex;gap:10px;align-items:center}
.gcc-ticker{
  display:flex;gap:6px;align-items:center;
  background:#111318;border:1px solid #252a38;border-radius:3px;
  padding:2px 8px;font-size:10px;
}
.gcc-ticker .sym{color:#e2e8f0;font-weight:500}
.gcc-ticker .px{color:#22c55e}
.gcc-ticker .chg{font-size:9px}
.gcc-right{display:flex;align-items:center;gap:10px}
.gcc-dot{width:6px;height:6px;border-radius:50%;background:#22c55e}
.gcc-live{font-size:10px;color:#4a5568;letter-spacing:.1em}
.gcc-time{font-size:10px;color:#4a5568}

/* -- Tabs -- */
[data-testid="stTabs"] [role="tablist"]{
  background:#0d0f14!important;
  border-bottom:1px solid #1d2130!important;
  gap:0!important;padding:0 20px!important;
  flex-wrap:nowrap!important;
}
[data-testid="stTabs"] [role="tab"]{
  font-family:'Geist Mono',monospace!important;
  font-size:11px!important;
  font-weight:400!important;
  text-transform:uppercase!important;
  letter-spacing:.08em!important;
  color:#4a5568!important;
  padding:8px 16px!important;
  border-bottom:2px solid transparent!important;
  border-radius:0!important;
  background:transparent!important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{
  color:#3b82f6!important;
  border-bottom-color:#3b82f6!important;
}
[data-testid="stTabs"] [role="tabpanel"]{
  padding:0!important;
}
/* remove the tab underline bar */
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{display:none!important}
[data-testid="stTabs"] [data-baseweb="tab-border"]{display:none!important}

/* -- Metrics strip -- */
[data-testid="metric-container"]{
  background:#0d0f14!important;
  border:none!important;
  border-right:1px solid #1d2130!important;
  border-radius:0!important;
  padding:6px 12px!important;
}
[data-testid="metric-container"] label{
  font-size:9px!important;color:#4a5568!important;
  text-transform:uppercase!important;letter-spacing:.12em!important;
  font-family:'Geist Mono',monospace!important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
  font-size:13px!important;font-weight:500!important;
  font-family:'Geist Mono',monospace!important;
  color:#e2e8f0!important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"]{
  font-size:10px!important;color:#4a5568!important;
}

/* -- Dividers -- */
hr{border-color:#1d2130!important;margin:0!important}

/* -- Dataframes -- */
[data-testid="stDataFrame"]{
  border:1px solid #1d2130!important;border-radius:3px!important;
}
[data-testid="stDataFrame"] th{
  background:#0d0f14!important;color:#4a5568!important;
  font-size:10px!important;text-transform:uppercase!important;
  letter-spacing:.08em!important;border-bottom:1px solid #1d2130!important;
  font-family:'Geist Mono',monospace!important;
}
[data-testid="stDataFrame"] td{
  font-size:11px!important;color:#e2e8f0!important;
  font-family:'Geist Mono',monospace!important;
  border-bottom:1px solid #111318!important;
}

/* -- Inputs / selects -- */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div{
  background:#111318!important;border-color:#252a38!important;
  border-radius:3px!important;font-size:11px!important;
  font-family:'Geist Mono',monospace!important;
}
.stSlider [data-baseweb="slider"] [role="progressbar"]{background:#3b82f6!important}
.stSlider [data-testid="stMarkdownContainer"] p{
  font-size:10px!important;color:#4a5568!important;
  font-family:'Geist Mono',monospace!important;
}

/* -- Radio -- */
[data-testid="stRadio"] label{
  font-size:11px!important;font-family:'Geist Mono',monospace!important;
}

/* -- Section headers -- */
.gcc-section{
  font-size:9px;color:#4a5568;letter-spacing:.14em;
  text-transform:uppercase;padding:10px 0 6px;
  border-bottom:1px solid #1d2130;margin-bottom:10px;
}
.gcc-badge{
  display:inline-block;padding:2px 10px;border-radius:3px;
  font-size:10px;font-weight:500;letter-spacing:.06em;
  font-family:'Geist Mono',monospace;
}

/* -- Column padding reset -- */
[data-testid="column"]{padding:0 4px!important}
[data-testid="stVerticalBlock"]{gap:4px!important}
[data-testid="stVerticalBlockBorderWrapper"]{gap:4px!important}
div.element-container{margin:0!important}
.stPlotlyChart{margin:0!important}

/* -- Plotly -- */
.js-plotly-plot .plotly .main-svg{background:transparent!important}
.stPlotlyChart{border:1px solid #1d2130;border-radius:3px}

/* -- Status bar -- */
.gcc-statusbar{
  display:flex;align-items:center;gap:20px;
  padding:0 20px;height:22px;
  background:#0d1a3a;border-top:1px solid #1d3f7a;
  font-size:10px;color:#60a5fa;
}
.gcc-statusbar span{color:#93c5fd;font-weight:500}

/* -- Regime badge colors -- */
.regime-calm{background:#052e16;border:1px solid #22c55e;color:#22c55e}
.regime-normal{background:#0c1a3a;border:1px solid #3b82f6;color:#3b82f6}
.regime-fearful{background:#2d0a0a;border:1px solid #ef4444;color:#ef4444}

/* -- Number inputs compact -- */
[data-testid="stNumberInput"] input{
  font-size:11px!important;padding:4px 8px!important;height:30px!important;
  font-family:'Geist Mono',monospace!important;
  background:#111318!important;border-color:#252a38!important;
}
[data-testid="stNumberInput"] button{height:30px!important;width:24px!important}
::-webkit-scrollbar{width:4px;background:#08090c}
::-webkit-scrollbar-thumb{background:#1d2130;border-radius:2px}
</style>
""", unsafe_allow_html=True)

# -- Imports --------------------------------------------------------------------
from data.fetcher          import fetch_ohlcv, get_latest_price, GCC_TICKERS, fetch_multiple
from engines.black_scholes import BlackScholes, implied_volatility
from engines.monte_carlo   import MonteCarlo
from engines.garch         import GARCHModel
from engines.portfolio     import PortfolioOptimiser
import utils.charts as charts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# -- Top bar -------------------------------------------------------------------
st.markdown(f"""
<div class="gcc-topbar">
  <div style="display:flex;align-items:center;gap:20px">
    <div class="gcc-logo">GCC <span>/</span> Derivatives Terminal</div>
    <div class="gcc-tickers">
      <div class="gcc-ticker"><span class="sym">AAPL</span><span class="px">$213.88</span><span class="chg" style="color:#22c55e">^1.24%</span></div>
      <div class="gcc-ticker"><span class="sym">^TASI</span><span class="px">11,842</span><span class="chg" style="color:#ef4444">v0.38%</span></div>
      <div class="gcc-ticker"><span class="sym">^DFMGI</span><span class="px">4,521</span><span class="chg" style="color:#22c55e">^0.71%</span></div>
    </div>
  </div>
  <div class="gcc-right">
    <div class="gcc-dot"></div>
    <div class="gcc-live">LIVE</div>
    <div class="gcc-time">GCC / MENA Markets</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -- Global controls (inline, compact) ----------------------------------------
with st.container():
    c = st.columns([1.2, 1, 1, 1, 1, 1, 1, 0.7])
    ticker_label = c[0].selectbox("Asset", list(GCC_TICKERS.keys()), index=3, label_visibility="collapsed")
    ticker = GCC_TICKERS[ticker_label]

    with st.spinner(""):
        try:
            S_live = get_latest_price(ticker)
        except Exception:
            S_live = 150.0

    S     = c[1].number_input("S", value=round(S_live, 2), format="%.2f", label_visibility="collapsed")
    K     = c[2].number_input("K", value=round(S_live, 2), format="%.2f", label_visibility="collapsed")
    T     = c[3].number_input("T (yrs)", value=0.25, min_value=0.01, max_value=3.0, step=0.01, format="%.2f", label_visibility="collapsed")
    r     = c[4].number_input("r", value=0.05, min_value=0.0, max_value=0.20, step=0.005, format="%.3f", label_visibility="collapsed")
    sigma = c[5].number_input("", value=0.25, min_value=0.01, max_value=1.5, step=0.01, format="%.2f", label_visibility="collapsed")
    opt_t = c[6].selectbox("Type", ["call","put"], label_visibility="collapsed")
    c[7].markdown(f"""
    <div style="padding:6px 0;font-size:9px;color:#4a5568;letter-spacing:.1em;line-height:1.8">
    S&nbsp;&nbsp;spot<br>K&nbsp;&nbsp;strike<br>T&nbsp;&nbsp;expiry<br>r&nbsp;&nbsp;rate&nbsp;&nbsp;&nbsp;&nbsp;vol
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# -- Tabs ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Options Pricer", "Monte Carlo", "Volatility", "Portfolio"
])

# +==============================================================================+
# |  TAB 1 - OPTIONS PRICER                                                    |
# +==============================================================================+
with tab1:
    bs       = BlackScholes(S, K, T, r, sigma)
    call_res = bs.compute_all("call")
    put_res  = bs.compute_all("put")
    active   = call_res if opt_t == "call" else put_res
    intrinsic = max(S-K,0) if opt_t=="call" else max(K-S,0)
    time_val  = max(active.price - intrinsic, 0)
    moneyness = "ATM" if abs(S-K)<K*0.02 else ("ITM" if (opt_t=="call" and S>K) or (opt_t=="put" and S<K) else "OTM")

    # metrics strip
    cols = st.columns(6)
    cols[0].metric(f"{opt_t.upper()} Price",  f"${active.price:.4f}")
    cols[1].metric("Intrinsic",               f"${intrinsic:.4f}")
    cols[2].metric("Time Value",              f"${time_val:.4f}")
    cols[3].metric("Moneyness",               moneyness)
    cols[4].metric("d1",                      f"{active.d1:.4f}")
    cols[5].metric("d2",                      f"{active.d2:.4f}")

    st.markdown("<hr>", unsafe_allow_html=True)

    left, mid, right = st.columns([1, 1.6, 1.2])

    with left:
        st.markdown('<div class="gcc-section">Greeks</div>', unsafe_allow_html=True)
        greeks_df = pd.DataFrame({
            "Greek": [" Delta", " Gamma", " Theta/day", " Vega/1%", " Rho/1%"],
            "Value": [
                f"{active.delta:+.5f}",
                f"{active.gamma:.6f}",
                f"{active.theta:+.5f}",
                f"{active.vega:.5f}",
                f"{active.rho:+.5f}",
            ],
        }).set_index("Greek")
        st.dataframe(greeks_df, use_container_width=True, height=210)

        st.markdown('<div class="gcc-section" style="margin-top:12px">Call vs Put</div>', unsafe_allow_html=True)
        comp_df = pd.DataFrame({
            "": ["Price","Delta","Theta","Rho"],
            "CALL":[f"{call_res.price:.4f}",f"{call_res.delta:+.4f}",f"{call_res.theta:+.5f}",f"{call_res.rho:+.4f}"],
            "PUT": [f"{put_res.price:.4f}", f"{put_res.delta:+.4f}", f"{put_res.theta:+.5f}", f"{put_res.rho:+.4f}"],
        }).set_index("")
        st.dataframe(comp_df, use_container_width=True, height=180)

        st.markdown('<div class="gcc-section" style="margin-top:12px">IV Solver</div>', unsafe_allow_html=True)
        mkt = st.number_input("Market price", value=float(round(active.price*1.05,4)), step=0.001, format="%.4f", label_visibility="collapsed")
        iv_sol = implied_volatility(mkt, S, K, T, r, opt_t)
        iv_col1, iv_col2 = st.columns(2)
        iv_col1.metric("Solved IV", f"{iv_sol*100:.2f}%" if not np.isnan(iv_sol) else "N/A")
        iv_col2.metric("vs Input ", f"{(iv_sol-sigma)*100:+.2f}%" if not np.isnan(iv_sol) else "-")

    with mid:
        st.markdown('<div class="gcc-section">IV Surface - Smile / Skew</div>', unsafe_allow_html=True)
        st.plotly_chart(charts.iv_surface_chart(S, sigma), use_container_width=True)

    with right:
        st.markdown('<div class="gcc-section">Delta Heatmap</div>', unsafe_allow_html=True)
        st.plotly_chart(charts.greeks_heatmap(S, K, r, sigma, opt_t), use_container_width=True)

        st.markdown('<div class="gcc-section" style="margin-top:8px">Scenario P&L</div>', unsafe_allow_html=True)
        scenarios = []
        for shock, label in [(0.05,"S+5%"),(-0.05,"S5%"),(0.10,"S+10%"),(-0.10,"S10%")]:
            bs2 = BlackScholes(S*(1+shock), K, T, r, sigma)
            pnl = bs2.price(opt_t) - active.price
            scenarios.append({"Scenario":label, "P&L":f"${pnl:+.3f}"})
        for vol_shock, label in [(0.05,"+5%"),(-0.05,"5%")]:
            bs2 = BlackScholes(S, K, T, r, max(sigma+vol_shock,0.01))
            pnl = bs2.price(opt_t) - active.price
            scenarios.append({"Scenario":label, "P&L":f"${pnl:+.3f}"})
        bs_1d = BlackScholes(S, K, max(T-1/365,0.001), r, sigma)
        pnl_1d = bs_1d.price(opt_t) - active.price
        scenarios.append({"Scenario":"1 day","P&L":f"${pnl_1d:+.3f}"})
        sc_df = pd.DataFrame(scenarios).set_index("Scenario")
        st.dataframe(sc_df, use_container_width=True, height=260)

# +==============================================================================+
# |  TAB 2 - MONTE CARLO                                                       |
# +==============================================================================+
with tab2:
    mc_col1, mc_col2 = st.columns([4, 1])
    with mc_col2:
        mc_paths = st.selectbox("Paths", [1000,5000,10000,25000], index=2)
        mc_T     = st.number_input("Horizon (yrs)", value=1.0, min_value=0.25, max_value=3.0, step=0.25)

    with st.spinner("Simulating"):
        mc = MonteCarlo(S0=S, mu=r, sigma=sigma, T=mc_T, N=mc_paths)
        mc_result = mc.simulate()
        stress    = mc.stress_test()

    cols = st.columns(6)
    cols[0].metric("95% VaR",   f"{mc_result.var_95*100:.2f}%")
    cols[1].metric("99% VaR",   f"{mc_result.var_99*100:.2f}%")
    cols[2].metric("95% CVaR",  f"{mc_result.cvar_95*100:.2f}%")
    cols[3].metric("99% CVaR",  f"{mc_result.cvar_99*100:.2f}%")
    cols[4].metric("E[Return]", f"{mc_result.mean_return*100:.2f}%")
    cols[5].metric("P(Loss)",   f"{mc_result.prob_loss*100:.1f}%")

    st.markdown("<hr>", unsafe_allow_html=True)

    p1, p2 = st.columns([3,2])
    with p1:
        st.markdown('<div class="gcc-section">GBM Price Paths</div>', unsafe_allow_html=True)
        st.plotly_chart(charts.mc_paths_chart(mc_result, S), use_container_width=True)
    with p2:
        st.markdown('<div class="gcc-section">Return Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(charts.mc_distribution_chart(mc_result), use_container_width=True)

    st.markdown('<div class="gcc-section" style="padding:10px 0 6px">Stress Scenarios</div>', unsafe_allow_html=True)
    stress_data = []
    for sr in stress:
        stress_data.append({
            "Scenario":      sr.scenario,
            "Shock":         f"{sr.shock*100:.0f}%",
            "Stressed Price":f"${sr.stressed_price:.2f}",
            "Total Loss":    f"{abs(sr.stressed_return)*100:.1f}%",
            "Stressed VaR":  f"{min((mc_result.var_95+abs(sr.shock))*100,100):.1f}%",
        })
    st.dataframe(pd.DataFrame(stress_data).set_index("Scenario"), use_container_width=True, height=220)

    st.markdown('<div class="gcc-section" style="padding:10px 0 6px">Terminal Price Percentiles</div>', unsafe_allow_html=True)
    pcts = [1,5,10,25,50,75,90,95,99]
    pct_df = pd.DataFrame(
        {"Percentile":[f"P{p}" for p in pcts],
         "Price":[f"${np.percentile(mc_result.final_prices,p):.2f}" for p in pcts]}
    ).set_index("Percentile").T
    st.dataframe(pct_df, use_container_width=True, height=100)

# +==============================================================================+
# |  TAB 3 - VOLATILITY                                                        |
# +==============================================================================+
with tab3:
    with st.spinner("Fitting GARCH(1,1)"):
        try:
            price_data  = fetch_ohlcv(ticker, period_years=3)
            garch_model = GARCHModel(price_data["Log_Returns"])
            g           = garch_model.fit_and_forecast(horizon=30)
            hist_dates  = price_data.index
        except Exception as e:
            st.error(f"GARCH error: {e}")
            st.stop()

    regime_cls = {"calm":"regime-calm","normal":"regime-normal","fearful":"regime-fearful"}[g.regime]
    st.markdown(f'<span class="gcc-badge {regime_cls}">> REGIME: {g.regime.upper()}</span>', unsafe_allow_html=True)

    cols = st.columns(5)
    cols[0].metric("Current Vol",  f"{g.current_vol*100:.2f}%")
    cols[1].metric("Long-run Vol", f"{g.long_run_vol*100:.2f}%")
    cols[2].metric("30d Hist Vol", f"{g.hist_vol_30d*100:.2f}%")
    cols[3].metric("1Y Hist Vol",  f"{g.hist_vol_252d*100:.2f}%")
    cols[4].metric("Persistence",  f"{g.persistence:.4f}")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.plotly_chart(charts.garch_forecast_chart(g, hist_dates), use_container_width=True)

    gl, gr = st.columns([1, 2])
    with gl:
        st.markdown('<div class="gcc-section">GARCH(1,1) Parameters</div>', unsafe_allow_html=True)
        param_df = pd.DataFrame({
            "Parameter": [" (omega)"," (alpha)"," (beta)","+","Long-run "],
            "Value":     [f"{g.omega:.6f}",f"{g.alpha:.4f}",f"{g.beta:.4f}",
                          f"{g.persistence:.4f}",f"{g.long_run_vol*100:.2f}%"],
        }).set_index("Parameter")
        st.dataframe(param_df, use_container_width=True)

    with gr:
        st.markdown('<div class="gcc-section">30-Day Forecast</div>', unsafe_allow_html=True)
        fc_df = pd.DataFrame({
            "Day": np.arange(1,31),
            "Forecast Vol (%)": np.round(g.forecast_vol*100,3),
            "Lower (%)":        np.round(g.lower_band*100,3),
            "Upper (%)":        np.round(g.upper_band*100,3),
        }).set_index("Day")
        st.dataframe(fc_df, use_container_width=True, height=310)

    # Price + vol overlay
    n_c  = len(g.conditional_vol)
    p_sub= price_data["Close"].iloc[-n_c:].values
    d_sub= price_data.index[-n_c:]
    fig_pv = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],vertical_spacing=0.04)
    fig_pv.add_trace(go.Scatter(x=d_sub,y=p_sub,mode="lines",line=dict(color=charts.BLUE,width=1.2),name="Price"),row=1,col=1)
    fig_pv.add_trace(go.Scatter(x=d_sub,y=g.conditional_vol*100,mode="lines",line=dict(color=charts.AMBER,width=1),name="Vol %"),row=2,col=1)
    fig_pv.update_layout(paper_bgcolor=charts.BG,plot_bgcolor=charts.BG,font=dict(family=charts.FONT,color=charts.TEXT),
        height=380,margin=dict(l=50,r=20,t=10,b=40),legend=dict(bgcolor=charts.BG,font=dict(color=charts.TEXT,size=10)),
        xaxis2=dict(gridcolor=charts.GRID),yaxis=dict(gridcolor=charts.GRID,title="Price"),
        yaxis2=dict(gridcolor=charts.GRID,title=" (%)"))
    st.plotly_chart(fig_pv, use_container_width=True)

# +==============================================================================+
# |  TAB 4 - PORTFOLIO                                                         |
# +==============================================================================+
with tab4:
    pf_col1, pf_col2, pf_col3 = st.columns([3,1,1])
    with pf_col2:
        portfolio_tickers = st.multiselect("Assets", list(GCC_TICKERS.values()),
            default=["AAPL","MSFT","AMZN","NVDA"], label_visibility="collapsed")
        if len(portfolio_tickers)<2: portfolio_tickers=["AAPL","MSFT","AMZN","NVDA"]
    with pf_col3:
        rf_pf = st.number_input("Rf %", value=5.0, step=0.5) / 100

    with st.spinner("Computing efficient frontier"):
        try:
            price_dict = fetch_multiple(portfolio_tickers, period_years=3)
            if len(price_dict)<2:
                st.warning("Need 2 assets."); st.stop()
            opt = PortfolioOptimiser(price_dict, risk_free=rf_pf, n_portfolios=500)
            fr  = opt.compute()
        except Exception as e:
            st.error(f"Portfolio error: {e}"); st.stop()

    ms, mv = fr.max_sharpe, fr.min_vol
    cols = st.columns(6)
    cols[0].metric("Max Sharpe",     f"{ms.sharpe:.3f}")
    cols[1].metric("Max SR Return",  f"{ms.exp_return*100:.2f}%")
    cols[2].metric("Max SR Vol",     f"{ms.volatility*100:.2f}%")
    cols[3].metric("Min Variance",   f"{mv.volatility*100:.2f}%")
    cols[4].metric("Min Var Return", f"{mv.exp_return*100:.2f}%")
    cols[5].metric("Assets",         str(len(fr.tickers)))

    st.markdown("<hr>", unsafe_allow_html=True)

    fl, fr_col = st.columns([2, 1])
    with fl:
        st.markdown('<div class="gcc-section">Efficient Frontier</div>', unsafe_allow_html=True)
        st.plotly_chart(charts.efficient_frontier_chart(fr), use_container_width=True)
        st.markdown('<div class="gcc-section" style="margin-top:6px">Correlation Matrix</div>', unsafe_allow_html=True)
        st.plotly_chart(charts.correlation_heatmap(fr.corr_matrix), use_container_width=True)

    with fr_col:
        st.markdown('<div class="gcc-section">Optimal Weights</div>', unsafe_allow_html=True)
        w_df = pd.DataFrame({
            "Asset":   fr.tickers,
            "MaxSR %": [f"{w*100:.1f}" for w in ms.weights],
            "MinVol %":[f"{w*100:.1f}" for w in mv.weights],
            "Ann.Ret": [f"{fr.mean_returns[t]*100:.2f}%" for t in fr.tickers],
        }).set_index("Asset")
        st.dataframe(w_df, use_container_width=True)

        st.markdown('<div class="gcc-section" style="margin-top:10px">Max Sharpe Weights</div>', unsafe_allow_html=True)
        colors_pie = [charts.BLUE,"#22c55e",charts.AMBER,charts.RED,"#8b5cf6","#06b6d4"]
        fig_pie = go.Figure(go.Pie(
            labels=fr.tickers, values=ms.weights, hole=0.55,
            marker=dict(colors=colors_pie[:len(fr.tickers)]),
            textfont=dict(color=charts.TEXT, family=charts.FONT, size=10),
        ))
        fig_pie.update_layout(paper_bgcolor=charts.BG,plot_bgcolor=charts.BG,
            font=dict(family=charts.FONT,color=charts.TEXT),height=260,
            margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(bgcolor=charts.BG,font=dict(color=charts.TEXT,size=10)))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="gcc-section" style="margin-top:10px">Asset Stats</div>', unsafe_allow_html=True)
        stats_df = pd.DataFrame({
            "Ann. Ret": (fr.mean_returns*100).map("{:.2f}%".format),
            "Ann. Vol": (np.sqrt(np.diag(fr.cov_matrix.values))*100).round(2),
        })
        st.dataframe(stats_df, use_container_width=True)

# -- Status bar ----------------------------------------------------------------
st.markdown("""
<div class="gcc-statusbar">
  <span style="color:#60a5fa">Model</span>&nbsp;<span>Black-Scholes 1973</span>
  &nbsp;&nbsp;<span style="color:#60a5fa">Data</span>&nbsp;<span>yfinance  15min delay</span>
  &nbsp;&nbsp;<span style="color:#60a5fa">Vol</span>&nbsp;<span>GARCH(1,1)</span>
  &nbsp;&nbsp;<span style="color:#60a5fa">MC Paths</span>&nbsp;<span>10,000 GBM</span>
  <span style="margin-left:auto;color:#1d3f7a">GCC Derivatives Terminal v1.0</span>
</div>
""", unsafe_allow_html=True)

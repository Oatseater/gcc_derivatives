# ⬡ GCC Derivatives Terminal

A Bloomberg-style quantitative finance dashboard for GCC equity derivatives.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Features

| Tab | Engine | What it does |
|-----|--------|-------------|
| **Options Pricer** | Black-Scholes (scratch) | Call/Put price, all 5 Greeks, IV solver, IV surface |
| **Monte Carlo** | GBM (10k paths) | VaR 95/99, CVaR, stress tests (2008, COVID, GCC) |
| **Volatility** | GARCH(1,1) | 30-day forecast, regime detection, confidence bands |
| **Portfolio** | Efficient Frontier | Max Sharpe, Min Vol, correlation matrix, weight pies |

---

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/gcc-derivatives.git
cd gcc-derivatives
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Run in Google Colab

```python
# Cell 1 — clone & install
!git clone https://github.com/YOUR_USERNAME/gcc-derivatives.git
%cd gcc-derivatives
!pip install -q -r requirements.txt pyngrok

# Cell 2 — launch
!python colab_runner.py
```

Click the **ngrok URL** printed in the output.

> **Tip:** Sign up for a free ngrok token at [ngrok.com](https://ngrok.com) and add  
> `ngrok.set_auth_token("YOUR_TOKEN")` in `colab_runner.py` for longer sessions.

---

## Deploy to Streamlit Cloud (free)

1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your fork, branch `main`, file `app.py`
4. Click **Deploy** — done in ~2 minutes

---

## Project Structure

```
gcc_derivatives/
├── app.py                   # Streamlit dashboard (4 tabs)
├── colab_runner.py          # Google Colab launcher
├── requirements.txt
├── engines/
│   ├── black_scholes.py     # BS pricer + Greeks + IV solver
│   ├── monte_carlo.py       # GBM simulation + VaR/CVaR
│   ├── garch.py             # GARCH(1,1) vol forecasting
│   └── portfolio.py         # Efficient frontier optimiser
├── data/
│   └── fetcher.py           # yfinance OHLCV fetcher + cache
└── utils/
    └── charts.py            # Plotly chart factory
```

---

## Finance Concepts

### Black-Scholes
Closed-form pricing of European options assuming log-normal returns, constant vol, and continuous trading. Greeks (Δ, Γ, θ, ν, ρ) measure sensitivity to each input parameter.

### Monte Carlo (GBM)
Simulate thousands of random price paths using `S(t+Δt) = S(t)·exp[(μ-σ²/2)Δt + σ√Δt·ε]`. Aggregate the terminal distribution for VaR and CVaR.

### GARCH(1,1)
`σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}` — captures volatility clustering where large moves follow large moves. Persistence `α+β` near 1 indicates long-memory vol.

### Efficient Frontier
Markowitz (1952): diversified portfolios on the frontier maximise return per unit of risk. We locate the Maximum Sharpe Ratio and Minimum Variance points.

---

## GCC Tickers

| Market | Yahoo Ticker |
|--------|-------------|
| Abu Dhabi ADX | `^FTFADGI` |
| Saudi Tadawul | `^TASI` |
| Dubai DFM | `^DFMGI` |
| Fallback US | `AAPL, MSFT, AMZN, NVDA` |

---

MIT License

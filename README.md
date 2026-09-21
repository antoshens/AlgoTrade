# AlgoTrade

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An algorithmic trading, portfolio optimization, and quantitative backtesting engine implemented in Python. AlgoTrade combines Modern Portfolio Theory (MPT), econometric volatility forecasting, and realistic market microstructure execution modeling.

---

## Overview

AlgoTrade provides tools for quantitative asset allocation, risk modeling, and strategy backtesting. The platform evaluates portfolios under realistic execution conditions, accounting for asset weight drift, broker commissions, and rebalancing turnover costs.

## Key Features

- **Portfolio Optimization & Modern Portfolio Theory**
  - Mean-Variance Optimization and Efficient Frontier generation.
  - Objective functions: Maximum Sharpe Ratio (tangency portfolio) and Maximum Sortino Ratio (downside risk optimization).
  - Covariance estimators: Sample Covariance, Ledoit-Wolf Shrinkage, and GARCH/EGARCH conditional volatility.
  - Expected returns models: Historical mean log returns and the Black-Litterman model incorporating market equilibrium priors and subjective views.
  - Turnover regularisation via L1 penalties (fixed or structural cost coupling) to limit portfolio churn.
  - Risk-free rate fetching from FRED (13-week T-Bills, 10-year Treasury Notes, and SOFR).

- **Econometric & Volatility Modeling**
  - Univariate GARCH(1,1) and EGARCH(1,1) volatility forecasting with Student's t-distribution for heavy-tailed financial returns.
  - Microstructure and proxy metrics: Parkinson volatility, Corwin-Schultz bid-ask spread estimators, and overnight gap analytics.

- **Out-of-Sample Backtesting Engine**
  - Walk-forward rolling simulation with configurable lookback windows and rebalancing schedules.
  - Execution realism: models transaction costs, turnover drag, and daily intra-period asset weight drift.
  - Benchmark comparison against equal-weighted allocations and the S&P 500 (`^GSPC`).

- **Performance & Risk Metrics**
  - Compound Annual Growth Rate (CAGR)
  - Maximum Drawdown (MDD)
  - Annualized Sharpe and Sortino Ratios
  - Calmar Ratio
  - Rolling risk-adjusted performance profiles

## Project Structure

```text
AlgoTrade/
├── data/                    # Market data acquisition and time-series preprocessing
│   ├── download_data.py     # Yahoo Finance data ingestion
│   ├── processors.py        # Returns, spreads, volatility, and market feature extraction
│   └── constants.py         # Trading constants and default parameters
├── src/
│   ├── portfolio/           # Optimization, Black-Litterman, and GARCH modules
│   │   ├── markowitz.py     # Mean-variance optimization and frontier routines
│   │   ├── black_litterman.py # Black-Litterman allocation model
│   │   └── garch.py         # ARCH/GARCH conditional volatility forecasting
│   └── backtest/            # Backtesting engine and performance analytics
│       ├── engine.py        # Walk-forward simulation and turnover execution
│       └── metrics.py       # Quantitative performance and drawdown metrics
├── notebooks/               # Research notebooks and exploratory analysis
│   ├── market_analysis/     # Return distributions, market memory, and volatility
│   ├── modern_portfolio_theory/ # Optimization and shrinkage experiments
│   └── models_backtesting/  # Strategy backtesting walkthroughs
├── pyproject.toml           # Package configuration
└── requirements.txt         # Project dependencies
```

## Installation

### Prerequisites
- Python 3.11 or higher
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/AlgoTrade.git
   cd AlgoTrade
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

### 1. Market Data Ingestion
```python
from datetime import datetime
from data.download_data import download_tickers_history

start = datetime(2020, 1, 1)
end = datetime(2024, 1, 1)
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

data = download_tickers_history(start_date=start, end_date=end, tickers=tickers)
```

### 2. Portfolio Optimization
```python
from src.portfolio import optimize_portfolio

# Calculate the efficient frontier using Ledoit-Wolf covariance shrinkage
frontier_df = optimize_portfolio(
    tickers_df=data,
    rf_base="T_BILLS",
    cov_model="LEDOIT_WOLF",
    returns_model="HISTORICAL",
    opt_type="BACKTEST",
    l1_coeff=0.01,
)

print(frontier_df.head())
```

### 3. Strategy Backtesting
```python
from src.backtest import perform_backtesting

results = perform_backtesting(
    tickers_df=data,
    init_portfolio_value=10000.0,
    optimization_metric="GARCH_SHARPE",
    returns_model="HISTORICAL",
    lookback_window=504,
    rebalancing_period=21,
    broker_commission=0.0005,
    l1_penalty_type="FIXED",
)

print(results[["portfolio_value", "equal_portfolio_value"]].tail())
```

## Research Notebooks

The `notebooks/` directory contains research workflows and exploratory analysis:
- **Market Analysis**: Statistical distribution testing, volatility clustering, and market memory.
- **Modern Portfolio Theory**: Empirical evaluations of sample covariance, Ledoit-Wolf shrinkage, and Black-Litterman models.
- **Models Backtesting**: Out-of-sample backtesting runs for GARCH- and EGARCH-based asset allocation strategies.

## Roadmap

- [ ] Local Large Language Model (LLM) integration to extract subjective views for the Black-Litterman model from financial news and filings.
- [ ] Non-linear transaction cost modeling and dynamic slippage estimation.
- [ ] Broker API integration for live execution.

## Disclaimer

This software is provided strictly for research and educational purposes. It does not constitute investment, financial, or trading advice.

## License

This project is licensed under the MIT License.

from datetime import datetime
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from data.processors import log_returns
from data.download_data import download_tickers_history

TRADING_DAYS_PER_YEAR = 252

RiskFreeRateBase = Literal['T_BILLS', 'TREASURY_NOTES', 'SOFR']
"""
The benchmark asset used to calculate the risk-free rate.

Options:
- 'T_BILLS': 13-week Treasury Bill (^IRX).
- 'TREASURY_NOTES': 10-year Treasury Note (^TNX).
- 'SOFR': Secured Overnight Financing Rate (SR3=F).
"""

OptimizationType = Literal['BACKTEST', 'LIVE', None]
"""
Determines the date range for fetching the risk-free rate.

Options:
- 'BACKTEST': Uses the start and end dates of the provided historical dataset.
- 'LIVE': Uses only the most recent date available in the dataset.
- None: Defaults to 'BACKTEST' behavior.
"""

@dataclass
class SharpeRatio:
    """
    Data structure to hold the results of the maximum Sharpe ratio optimization.

    Attributes
    ----------
    max_sharpe : float
        The maximum Sharpe ratio achieved.
    tangency_return : float
        The expected annualized return of the tangency portfolio.
    tangency_vol : float
        The expected annualized volatility of the tangency portfolio.
    """
    max_sharpe: float
    tangency_return: float
    tangency_vol: float

def get_risk_free_rate(
        base: RiskFreeRateBase,
        start_date: datetime,
        end_date: datetime) -> float:
    """
    Fetches the annualized risk-free rate for a given period and benchmark.

    Parameters
    ----------
    base : RiskFreeRateBase
        The benchmark to use for the risk-free rate (e.g., 'T_BILLS', 'TREASURY_NOTES', 'SOFR').
    start_date : datetime
        The start date for fetching the historical rate.
    end_date : datetime
        The end date for fetching the historical rate.

    Returns
    -------
    float
        The most recent annualized risk-free rate as a decimal (e.g., 0.05 for 5%).
    
    Raises
    ------
    ValueError
        If an unknown RiskFreeRateBase literal is provided.
    """
    ticker: str
    match base:
        case 'T_BILLS':
            ticker = '^IRX'
        case 'TREASURY_NOTES':
            ticker = '^TNX'
        case 'SOFR':
            ticker = 'SR3=F'
        case _: raise ValueError(f'Unknown literal for base {base}')

    risk_ticker_df = download_tickers_history(start_date, end_date, [ticker])

    current_rf_annual = risk_ticker_df.iloc[-1] / 100
    risk_free_rate = current_rf_annual.loc[ticker]['Close']
    return risk_free_rate

def gmf_return_optimization(tickers_df: pd.DataFrame, rf_base: RiskFreeRateBase, opt_type: OptimizationType = None) -> pd.DataFrame:
    """
    Calculates the efficient frontier by minimizing volatility for various target returns.

    Parameters
    ----------
    tickers_df : pd.DataFrame
        Historical price data for the assets, expected to contain 'Close' prices.
    rf_base : RiskFreeRateBase
        The benchmark to use for calculating the risk-free rate.
    opt_type : OptimizationType, optional
        Determines the date range for the risk-free rate based on backtest or live mode.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents a portfolio on the efficient frontier, 
        containing its weights, return, volatility, and Sharpe ratio.
        
    Raises
    ------
    ValueError
        If an unrecognized opt_type parameter value is provided.
    """
    # Calculate optimization params
    num_assets = tickers_df.columns.levels[0].nunique()
    log_ret = log_returns(tickers_df)
    yr_cov = log_ret.cov() * TRADING_DAYS_PER_YEAR  # type: ignore
    init_weights = np.ones(num_assets) / num_assets
    expected_returns = log_ret.mean() * TRADING_DAYS_PER_YEAR

    match opt_type:
        case 'BACKTEST' | None:
            rf_start_date = pd.to_datetime(tickers_df.index[0])
            rf_end_date = pd.to_datetime(tickers_df.index[-1])
        case 'LIVE':
            rf_start_date = pd.to_datetime(tickers_df.index[-1])
            rf_end_date = pd.to_datetime(tickers_df.index[-1])
        case _: raise ValueError(f'Unrecognized param value: {opt_type}.')

    risk_free_rate = get_risk_free_rate(rf_base, rf_start_date, rf_end_date)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    ]

    # Bounds: Long-only (0 <= w_i <= 1)
    bounds = tuple((0, 0.4) for _ in range(num_assets))

    no_target_opt = minimize(
        fun=lambda weights: 0.5 * (weights.T @ yr_cov @ weights),
        x0=init_weights,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds
    )

    # Constraints over the possible target_returns
    min_target = np.dot(no_target_opt.x, expected_returns)
    max_target = expected_returns.max()
    target_returns = np.linspace(min_target - 1e-5, max_target + 1e-5, 100)

    # Optimize with the target_return constraints
    optimum_results = []
    last_optimal_weights = init_weights
    for target_return in target_returns:    
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return},
        ]

        res = minimize(
                fun=lambda weights: 0.5 * (weights.T @ yr_cov @ weights),
                x0=last_optimal_weights,
                method='SLSQP',
                constraints=constraints,
                bounds=bounds
            )

        if res.success:
            last_optimal_weights = res.x
            ret = np.dot(last_optimal_weights, expected_returns)
            vol = np.sqrt(last_optimal_weights.T @ yr_cov @ last_optimal_weights)
            sharpe = (ret - risk_free_rate) / vol

            optimum_results.append({
                'weights': last_optimal_weights,
                'return': ret,
                'vol': vol,
                'sharpe': sharpe,
            })

    return pd.DataFrame(optimum_results)

def max_sharpe_objective(weights, expected_returns, risk_free_rate: float, yr_cov):
    """
    Objective function to find the maximum Sharpe ratio (returns negative Sharpe ratio for minimization).

    Parameters
    ----------
    weights : np.ndarray
        Array of portfolio weights.
    expected_returns : np.ndarray
        Array of expected annualized returns for the assets.
    risk_free_rate : float
        The annualized risk-free rate.
    yr_cov : pd.DataFrame
        The annualized covariance matrix of the assets.

    Returns
    -------
    float
        The negative Sharpe ratio of the portfolio.
    """
    ret = np.dot(weights, expected_returns)
    vol = np.sqrt(weights.T @ yr_cov @ weights)

    # negative Sharpe ratio so minimize() finds the maximum
    return -(ret - risk_free_rate) / vol

def find_max_sharpe(tickers_df: pd.DataFrame, rf_base: RiskFreeRateBase, opt_type: OptimizationType = None) -> tuple[SharpeRatio, pd.DataFrame]:
    """
    Optimizes portfolio weights to maximize the Sharpe ratio (find the tangency portfolio).

    Parameters
    ----------
    tickers_df : pd.DataFrame
        Historical price data for the assets.
    rf_base : RiskFreeRateBase
        The benchmark to use for calculating the risk-free rate.
    opt_type : OptimizationType, optional
        Determines the date range for the risk-free rate based on backtest or live mode.

    Returns
    -------
    tuple[SharpeRatio, pd.DataFrame]
        A tuple containing:
        - SharpeRatio object with tangency portfolio metrics.
        - DataFrame containing the percentage weights for each stock in the optimal portfolio.
        
    Raises
    ------
    ValueError
        If an unrecognized opt_type parameter value is provided.
    RuntimeError
        If the scipy SLSQP optimizer fails to find a solution.
    """
    # Calculate optimization params
    num_assets = tickers_df.columns.levels[0].nunique()
    log_ret = log_returns(tickers_df)
    yr_cov = log_ret.cov() * TRADING_DAYS_PER_YEAR  # type: ignore
    init_weights = np.ones(num_assets) / num_assets
    expected_returns = log_ret.mean() * TRADING_DAYS_PER_YEAR
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 0.4) for _ in range(num_assets))

    match opt_type:
        case 'BACKTEST' | None:
            rf_start_date = pd.to_datetime(tickers_df.index[0])
            rf_end_date = pd.to_datetime(tickers_df.index[-1])
        case 'LIVE':
            rf_start_date = pd.to_datetime(tickers_df.index[-1])
            rf_end_date = pd.to_datetime(tickers_df.index[-1])
        case _: raise ValueError(f'Unrecognized param value: {opt_type}.')
    
    risk_free_rate = get_risk_free_rate(rf_base, rf_start_date, rf_end_date)

    # Find the Sharpe Ratio optimum
    res_max_sharpe = minimize(
        fun=lambda w: max_sharpe_objective(
            w,
            expected_returns=expected_returns,
            risk_free_rate=risk_free_rate,
            yr_cov=yr_cov),
        x0=init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if res_max_sharpe.success:
        optimal_weights = res_max_sharpe.x
        exact_max_ret = np.dot(optimal_weights, expected_returns)
        exact_max_vol = np.sqrt(optimal_weights.T @ yr_cov @ optimal_weights)
        exact_max_sharpe = (exact_max_ret - risk_free_rate) / exact_max_vol

        max_sharpe_stocks_weights = pd.DataFrame()
        tickers = tickers_df.columns.get_level_values(0).unique()
        for i, (name) in enumerate(tickers):
            max_sharpe_stocks_weights[name] = [round(optimal_weights[i], 2) * 100]

        max_sharpe = SharpeRatio(
            max_sharpe=exact_max_sharpe,
            tangency_return=exact_max_ret,
            tangency_vol=exact_max_vol,
        )
    else: raise RuntimeError('The optimizator finished work with an error or was .')

    return (max_sharpe, max_sharpe_stocks_weights)

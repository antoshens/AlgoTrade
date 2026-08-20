from datetime import datetime
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from data.processors import log_returns
from sklearn.covariance import LedoitWolf

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

OptimizationModel = Literal[
    'CLASSIC',
    'LEDOIT_WOLF',
    'SORTINO',
    'BLACK_LITTERMAN',
    'GARCH',
    'DCC_GARCH',
]
"""
Covariance estimation or portfolio optimization model to use.

Options:
- 'CLASSIC': Sample covariance matrix scaled by trading days.
- 'LEDOIT_WOLF': Ledoit-Wolf shrinkage covariance estimator.
- 'SORTINO': Downside deviation (semi-variance) based optimization.
- 'BLACK_LITTERMAN': Black-Litterman model with market views.
- 'GARCH': Univariate GARCH volatility modeling.
- 'DCC_GARCH': Dynamic Conditional Correlation GARCH model.
"""

@dataclass
class SharpeRatio:
    """
    Data structure holding the results of the Maximum Sharpe Ratio (Tangency Portfolio) optimization.

    Attributes
    ----------
    max_sharpe : float
        The maximum annualized Sharpe ratio achieved: (Return - RiskFreeRate) / Volatility.
    tangency_return : float
        The expected annualized return of the tangency portfolio.
    tangency_vol : float
        The expected annualized total volatility (standard deviation) of the tangency portfolio.
    """
    max_sharpe: float
    tangency_return: float
    tangency_vol: float

@dataclass
class SortinoRatio:
    """
    Data structure holding the results of the Maximum Sortino Ratio optimization.

    Attributes
    ----------
    max_sortino : float
        The maximum annualized Sortino ratio achieved: (Return - RiskFreeRate) / DownsideVolatility.
    tangency_return : float
        The expected annualized return of the tangency portfolio.
    tangency_vol : float
        The expected annualized downside volatility (semi-deviation below daily risk-free rate).
    """
    max_sortino: float
    tangency_return: float
    tangency_vol: float

def get_risk_free_rate(
    base: RiskFreeRateBase,
    start_date: datetime,
    end_date: datetime,
    opt_type: OptimizationType = None
) -> float:
    """
    Fetches and calculates the annualized risk-free rate for a given period and benchmark from FRED.

    Parameters
    ----------
    base : RiskFreeRateBase
        The benchmark asset used for the risk-free rate:
        - 'T_BILLS': 13-week Treasury Bill (FRED: 'DGS3MO').
        - 'TREASURY_NOTES': 10-year Treasury Note (FRED: 'DGS10').
        - 'SOFR': Secured Overnight Financing Rate (FRED: 'SOFR').
    start_date : datetime
        The start date for fetching the historical rate series.
    end_date : datetime
        The end date for fetching the historical rate series.
    opt_type : OptimizationType, optional
        Determines how the rate is sampled:
        - 'BACKTEST' or None: Computes the mean annualized rate over the [start_date, end_date] window.
        - 'LIVE': Uses the single most recent available rate observation.

    Returns
    -------
    float
        The annualized risk-free rate expressed as a decimal (e.g., 0.045 for 4.5%).
    
    Raises
    ------
    ValueError
        If an unknown `base` or `opt_type` literal is provided, or if FRED returns no data.
    """
    ticker: str
    match base:
        case 'T_BILLS':
            ticker = 'DGS3MO'
        case 'TREASURY_NOTES':
            ticker = 'DGS10'
        case 'SOFR':
            ticker = 'SOFR'
        case _: raise ValueError(f'Unknown literal for base {base}')

    # Download the FRED data from the open source
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}'
    risk_ticker_df = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.').loc[start_date:end_date].dropna()
    if risk_ticker_df.empty:
        raise ValueError(
            f"No data returned from FRED for ticker '{ticker}' between {start_date.date()} and {end_date.date()}."
        )

    risk_free_rate = 0.0
    match opt_type:
        case 'BACKTEST' | None:
            risk_free_rate = float(risk_ticker_df.mean().iloc[0])
        case 'LIVE':
            risk_free_rate = float(risk_ticker_df.iloc[-1, 0]) # type: ignore
        case _: raise ValueError(f'Unknown literal for opt_type {opt_type}')

    return risk_free_rate / 100

def optimize_portfolio(
    tickers_df: pd.DataFrame,
    rf_base: RiskFreeRateBase,
    opt_models: list[OptimizationModel] = ['CLASSIC'],
    opt_type: OptimizationType = None
) -> pd.DataFrame:
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
    num_assets = tickers_df.columns.levels[0].nunique() # type: ignore
    log_ret = log_returns(tickers_df)
    yr_cov = np.array(num_assets) # assets coveriance matrix
    init_weights = np.ones(num_assets) / num_assets
    expected_returns = np.array(log_ret.mean(axis=0) * TRADING_DAYS_PER_YEAR)

    # Defining optimization type
    match opt_type:
        case 'BACKTEST' | None:
            rf_start_date = pd.to_datetime(tickers_df.index[0]).tz_localize(None).to_pydatetime()
            rf_end_date = pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
        case 'LIVE':
            rf_start_date = pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
            rf_end_date = pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
        case _: raise ValueError(f'Unrecognized opt_value param value: {opt_type}.')

    risk_free_rate = get_risk_free_rate(rf_base, rf_start_date, rf_end_date, opt_type)

    for model in opt_models:
        match model:
            case 'CLASSIC':
                yr_cov = np.array(log_ret.cov() * TRADING_DAYS_PER_YEAR)  # type: ignore
            case 'LEDOIT_WOLF':
                lw = LedoitWolf()
                lw.fit(log_ret.to_xarray())
                yr_cov = np.array(lw.covariance_ * TRADING_DAYS_PER_YEAR)
            case _: raise ValueError(f'Unrecognized opt_model param value: {model}.')

    # Optimization
    optimum_results = _run_portfolio_optimization(yr_cov, init_weights, expected_returns, risk_free_rate)

    return pd.DataFrame(optimum_results)

def _run_portfolio_optimization(
    yr_cov: np.ndarray,
    init_weights: np.ndarray,
    expected_returns: np.ndarray,
    risk_free_rate: float,
) -> list:
    """Internal helper that performs the SLSQP portfolio optimization loop."""
    num_assets = yr_cov.__len__()
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

    return optimum_results

def find_max_sharpe(
    tickers_df: pd.DataFrame,
    rf_base: RiskFreeRateBase,
    opt_models: list[OptimizationModel] = ['CLASSIC'],
    opt_type: OptimizationType = None
) -> tuple[SharpeRatio, pd.DataFrame]:
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
    num_assets = tickers_df.columns.levels[0].nunique() # type: ignore
    log_ret = log_returns(tickers_df)
    yr_cov = np.array(num_assets) # assets covariance matrix
    init_weights = np.ones(num_assets) / num_assets
    expected_returns = np.array(log_ret.mean(axis=0) * TRADING_DAYS_PER_YEAR)

    match opt_type:
        case 'BACKTEST' | None:
            rf_start_date = pd.to_datetime(tickers_df.index[0]).tz_localize(None).to_pydatetime()
            rf_end_date = pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
        case 'LIVE':
            rf_start_date = pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
            rf_end_date = pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
        case _: raise ValueError(f'Unrecognized opt_value param value: {opt_type}.')
    
    risk_free_rate = get_risk_free_rate(rf_base, rf_start_date, rf_end_date, opt_type)

    for model in opt_models:
        match model:
            case 'CLASSIC':
                yr_cov = np.array(log_ret.cov() * TRADING_DAYS_PER_YEAR)  # type: ignore
            case 'LEDOIT_WOLF':
                lw = LedoitWolf()
                lw.fit(np.array(log_ret.values))
                yr_cov = np.array(lw.covariance_ * TRADING_DAYS_PER_YEAR)
            case _: raise ValueError(f'Unrecognized opt_model param value: {model}.')

    # Find the Sharpe Ratio optimum
    return _maximize_sharpe_ratio(tickers_df, init_weights, yr_cov, expected_returns, risk_free_rate)

def _max_sharpe_objective(
        weights: np.ndarray,
        expected_returns: np.ndarray,
        risk_free_rate: float,
        yr_cov: np.ndarray) -> float:
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

def _maximize_sharpe_ratio(
    tickers_df: pd.DataFrame | pd.Series,
    init_weights: np.ndarray,
    yr_cov: np.ndarray,
    expected_returns: np.ndarray,
    risk_free_rate: float
) -> tuple[SharpeRatio, pd.DataFrame]:
    """Internal helper that performs the SLSQP Sharpe ratio optimization."""
    num_assets = yr_cov.__len__()
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 0.4) for _ in range(num_assets))

    res_max_sharpe = minimize(
        fun=lambda w: _max_sharpe_objective(
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
    else: raise RuntimeError('The optimizator finished work with an error or was aborted.')

    return (max_sharpe, max_sharpe_stocks_weights)

def find_max_sortino(
    tickers_df: pd.DataFrame,
    rf_base: RiskFreeRateBase,
    opt_type: OptimizationType = None
) -> tuple[SortinoRatio, pd.DataFrame]:
    """
    Optimizes portfolio weights to maximize the Sortino ratio (find the tangency portfolio).

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
    tuple[SortinoRatio, pd.DataFrame]
        A tuple containing:
        - SortinoRatio object with tangency portfolio metrics.
        - DataFrame containing the percentage weights for each stock in the optimal portfolio.
        
    Raises
    ------
    ValueError
        If an unrecognized opt_type parameter value is provided.
    RuntimeError
        If the scipy SLSQP optimizer fails to find a solution.
    """
    # Calculate optimization params
    num_assets = tickers_df.columns.levels[0].nunique() # type: ignore
    log_ret = np.array(log_returns(tickers_df))
    init_weights = np.ones(num_assets) / num_assets
    expected_returns = np.array(log_ret.mean(axis=0) * TRADING_DAYS_PER_YEAR)

    match opt_type:
        case 'BACKTEST' | None:
            rf_start_date = pd.to_datetime(tickers_df.index[0]).tz_localize(None).to_pydatetime()
            rf_end_date = pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
        case 'LIVE':
            rf_start_date = pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
            rf_end_date = pd.to_datetime(tickers_df.index[-1]).tz_localize(None).to_pydatetime()
        case _: raise ValueError(f'Unrecognized opt_value param value: {opt_type}.')
    
    risk_free_rate = get_risk_free_rate(rf_base, rf_start_date, rf_end_date, opt_type)

    # Find the Sortino Ratio optimum
    return _maximize_sortino_ratio(tickers_df, init_weights, log_ret, expected_returns, risk_free_rate)

def _max_sortino_objective(
    weights: np.ndarray,
    log_returns: np.ndarray,
    expected_returns: np.ndarray,
    risk_free_rate: float) -> float:
    """
    Objective function to find the maximum Sortino ratio (returns negative Sortino ratio for minimization).

    Parameters
    ----------
    weights : np.ndarray
        Array of portfolio weights.
    log_returns: np.ndarray
        Array of daily logarithmic assets returns
    expected_returns : np.ndarray
        Array of expected annualized returns for the assets.
    risk_free_rate : float
        The annualized risk-free rate.

    Returns
    -------
    float
        The negative Sortino ratio of the portfolio.
    """
    ret = np.dot(weights, expected_returns)

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    square_negative_deviations = np.minimum(0, weights @ log_returns.T - daily_rf) ** 2
    downside_vol = np.sqrt((square_negative_deviations).mean(axis=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)

    # negative Sortino ratio so minimize() finds the maximum
    return -(ret - risk_free_rate) / downside_vol

def _maximize_sortino_ratio(
    tickers_df: pd.DataFrame | pd.Series,
    init_weights: np.ndarray,
    log_returns: np.ndarray,
    expected_returns: np.ndarray,
    risk_free_rate: float
) -> tuple[SortinoRatio, pd.DataFrame]:
    """Internal helper that performs the SLSQP Sortino ratio optimization."""
    num_assets = tickers_df.columns.levels[0].nunique() # type: ignore
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 0.4) for _ in range(num_assets))

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    square_negative_deviations = np.minimum(0, log_returns - daily_rf) ** 2

    res_max_sortino = minimize(
        fun=lambda w: _max_sortino_objective(
            w,
            log_returns,
            expected_returns=expected_returns,
            risk_free_rate=risk_free_rate),
        x0=init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if res_max_sortino.success:
        optimal_weights = res_max_sortino.x
        exact_max_ret = np.dot(optimal_weights, expected_returns)

        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
        square_negative_deviations = np.minimum(0, optimal_weights @ log_returns.T - daily_rf) ** 2
        exact_max_negative_vol = np.sqrt((square_negative_deviations).mean(axis=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        exact_max_sortino = (exact_max_ret - risk_free_rate) / exact_max_negative_vol

        max_sortino_stocks_weights = pd.DataFrame()
        tickers = tickers_df.columns.get_level_values(0).unique()
        for i, (name) in enumerate(tickers):
            max_sortino_stocks_weights[name] = [round(optimal_weights[i], 2) * 100]

        max_sortino = SortinoRatio(
            max_sortino=exact_max_sortino,
            tangency_return=exact_max_ret,
            tangency_vol=exact_max_negative_vol,
        )
    else: raise RuntimeError('The optimizator finished work with an error or was aborted.')

    return (max_sortino, max_sortino_stocks_weights)

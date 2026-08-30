from typing import TypeVar

import numpy as np
import pandas as pd

PandasData = TypeVar("PandasData", pd.DataFrame, pd.Series)


def log_returns(stocks: PandasData) -> PandasData:
    """Calculate logarithmic daily returns for stock price data.

    Parameters
    ----------
    stocks : pd.DataFrame | pd.Series
        Price data containing 'Close' prices or MultiIndex ticker history.

    Returns
    -------
    pd.DataFrame | pd.Series
        Daily log returns indexed by Date.

    Raises
    ------
    ValueError
        If stocks input is None.
    """
    if stocks is None:
        raise ValueError("The stocks input DataFrame is None.")

    if isinstance(stocks, pd.DataFrame) and isinstance(stocks.columns, pd.MultiIndex):
        level_name = "Price" if "Price" in stocks.columns.names else 1
        close_prices = stocks.xs("Close", axis=1, level=level_name)

    elif isinstance(stocks, pd.DataFrame) and "Close" in stocks.columns:
        close_prices = stocks["Close"]

    else:
        close_prices = stocks

    log_returns = np.log(close_prices).diff().dropna()  # type: ignore

    return log_returns


def overnight_gaps_prc(stocks: pd.DataFrame) -> pd.DataFrame:
    """Calculate the overnight price gap percentage between previous close and current open prices.

    Parameters
    ----------
    stocks : pd.DataFrame
        Price data containing 'Open' and 'Close' prices.

    Returns
    -------
    pd.DataFrame
        Overnight price gaps in percentage (%).

    Raises
    ------
    ValueError
        If stocks input is None or missing required 'Open' and 'Close' columns.
    """
    if stocks is None:
        raise ValueError("The stocks input DataFrame is None.")

    if isinstance(stocks, pd.DataFrame) and isinstance(stocks.columns, pd.MultiIndex):
        level_name = "Price" if "Price" in stocks.columns.names else 1
        open_prices = stocks.xs("Open", axis=1, level=level_name)
        close_prices = stocks.xs("Close", axis=1, level=level_name)

    elif isinstance(stocks, pd.DataFrame) and {"Open", "Close"}.issubset(
        stocks.columns
    ):
        close_prices = stocks["Close"]
        open_prices = stocks["Open"]

    else:
        raise ValueError(
            "The stocks input must be a DataFrame containing both 'Open' and 'Close' prices."
        )

    overnight_gap = np.log(open_prices / close_prices.shift(1)) * 100

    return overnight_gap.dropna()  # type: ignore


def rolling_overnight_gaps_std(stocks: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate the rolling standard deviation of overnight price gaps over a specified window.

    Parameters
    ----------
    stocks : pd.DataFrame
        Price data containing 'Open' and 'Close' prices.
    window : int
        Rolling window size in days.

    Returns
    -------
    pd.DataFrame
        Rolling standard deviation of overnight gaps.
    """
    overnight_gap = overnight_gaps_prc(stocks)

    return overnight_gap.rolling(window).std()


def intraday_returns_prc(stocks: pd.DataFrame) -> pd.DataFrame:
    """Calculate the intraday percentage log return between open and close prices.

    Parameters
    ----------
    stocks : pd.DataFrame
        Price data containing 'Open' and 'Close' prices.

    Returns
    -------
    pd.DataFrame
        Intraday percentage log returns (%).

    Raises
    ------
    ValueError
        If stocks input is None or missing required 'Open' and 'Close' columns.
    """
    if stocks is None:
        raise ValueError("The stocks input DataFrame is None.")

    if isinstance(stocks, pd.DataFrame) and isinstance(stocks.columns, pd.MultiIndex):
        level_name = "Price" if "Price" in stocks.columns.names else 1
        open_prices = stocks.xs("Open", axis=1, level=level_name)
        close_prices = stocks.xs("Close", axis=1, level=level_name)

    elif isinstance(stocks, pd.DataFrame) and {"Open", "Close"}.issubset(
        stocks.columns
    ):
        close_prices = stocks["Close"]
        open_prices = stocks["Open"]

    else:
        raise ValueError(
            "The stocks input must be a DataFrame containing both 'Open' and 'Close' prices."
        )

    intraday_return = np.log(close_prices / open_prices) * 100

    return intraday_return.dropna()  # type: ignore


def daily_spread_pct(stocks: pd.DataFrame) -> pd.DataFrame:
    """Calculate the relative daily high-low price spread as a percentage of the close price.

    Parameters
    ----------
    stocks : pd.DataFrame
        Price data containing 'High', 'Low', and 'Close' prices.

    Returns
    -------
    pd.DataFrame
        Relative daily high-low price spread (%).

    Raises
    ------
    ValueError
        If stocks input is None or missing required 'High', 'Low', and 'Close' columns.
    """
    if stocks is None:
        raise ValueError("The stocks input DataFrame is None.")

    if isinstance(stocks, pd.DataFrame) and isinstance(stocks.columns, pd.MultiIndex):
        level_name = "Price" if "Price" in stocks.columns.names else 1
        close_prices = stocks.xs("Close", axis=1, level=level_name)
        low_prices = stocks.xs("Low", axis=1, level=level_name)
        high_prices = stocks.xs("High", axis=1, level=level_name)

    elif isinstance(stocks, pd.DataFrame) and {"Low", "High", "Close"}.issubset(
        stocks.columns
    ):
        close_prices = stocks["Close"]
        low_prices = stocks["Low"]
        high_prices = stocks["High"]

    else:
        raise ValueError(
            "The stocks input must be a DataFrame containing 'Low', 'High' and 'Close' prices."
        )

    intraday_spread = ((high_prices - low_prices) / close_prices) * 100

    return intraday_spread.dropna()  # type: ignore


def rolling_daily_spreads_mean(stocks: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate the rolling mean of daily high-low price spreads over a specified window.

    Parameters
    ----------
    stocks : pd.DataFrame
        Price data containing 'High', 'Low', and 'Close' prices.
    window : int
        Rolling window size in days.

    Returns
    -------
    pd.DataFrame
        Rolling mean of high-low price spreads.
    """
    intraday_spread = daily_spread_pct(stocks).dropna()

    return intraday_spread.rolling(window).mean()


def rolling_vol_daily(stocks: PandasData, window: int) -> PandasData:
    """Calculate rolling daily volatility (standard deviation of log returns) over a specified window.

    Parameters
    ----------
    stocks : pd.DataFrame | pd.Series
        Price data containing stock prices.
    window : int
        Rolling window size in days.

    Returns
    -------
    pd.DataFrame | pd.Series
        Rolling daily volatility indexed by Date.
    """
    returns = log_returns(stocks)
    return returns.rolling(window).std().dropna()

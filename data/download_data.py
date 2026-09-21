from datetime import datetime

import pandas as pd
import yfinance as yf


def download_tickers_history(
    start_date: datetime, end_date: datetime, tickers: list[str]
) -> pd.DataFrame:
    """
    Downloads the selected ticker price history data from Yahoo Finance engine.

    Returns
    -------
    Open : Price at market open
    High : Highest price during the trading session
    Low : Lowest price during the trading session
    Close : Price at market close
    Volume : Number of shares traded
    """

    # download market data for a list of tickers
    tckrs_history = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    return tckrs_history if tckrs_history is not None else pd.DataFrame()

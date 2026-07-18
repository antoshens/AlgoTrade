from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict
import pandas as _pd
import yfinance as yf

@dataclass
class TickerHistory(TypedDict):
    name: str
    data: _pd.DataFrame

def download_tickers_history(start_date: datetime, end_date: datetime, tickers: list[str]) -> list[TickerHistory]:   
    tckrs_history = [download_single_ticket(start_date, end_date, t) for t in tickers]

    return tckrs_history;

def download_single_ticket(start_date: datetime, end_date: datetime, ticker: str) -> TickerHistory:
    """
        Downloads the selected ticker price history data from Yahoo Finance engine.
        
        The Data contains following fields: \n
        **Open**: Price at market open \n
        **High**: Highest price during the trading session \n
        **Low**: Lowest price during the trading session \n
        **Close**: Price at market close \n
        **Volume**: Number of shares traded \n
    """

    # download market data for a list of tickers
    tckr_history = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False
    )

    return TickerHistory(name=ticker, data=tckr_history or _pd.DataFrame())

from typing import Literal

from pydantic import BaseModel, Field


class AssetView(BaseModel):
    """
    Data contract between AI agent and Black-Littermann core
    """

    view_type: Literal["ABSOLUTE", "RELATIVE"]
    base_asset: str = Field(description="Target asset's ticker (f. e. 'NVDA')")
    target_asset: str | None = Field(
        None, description="Asset's ticker for RELATIVE view (f. e. 'MSFT')"
    )
    expected_outperformance: float = Field(
        description="Expected annual return or split returns diff, for example 0.05 for +5%"
    )
    confidence: float = Field(
        ge=0.01,
        le=1.0,
        description="Degree of analyst confidence from 0.01 (very weak) to 1.0 (absolute)",
    )
    rationale: str = Field(
        description="Brief fundamental rationale for the news/reporting view"
    )


class MarketViewsReport(BaseModel):
    as_of_date: str
    views: list[AssetView]

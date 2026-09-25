from typing import Literal

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("expected_outperformance")
    @classmethod
    def clamp_returns(cls, v: float) -> float:
        # Hard safeguard against extreme figures
        if abs(v) > 0.20:
            raise ValueError(
                f"Expected outperformance {v} exceeds reasonable ±20% annual bounds."
            )
        return v


class MarketViewsReport(BaseModel):
    as_of_date: str = Field(description="Date to which views are formed (YYYY-MM-DD)")
    views: list[AssetView] = Field(
        default_factory=list, description="List of formed views"
    )
    market_sentiment_summary: str = Field(
        description="Summary of macro and market background"
    )

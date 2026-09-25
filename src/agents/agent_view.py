import json
from collections.abc import Sequence

import pandas as pd
from openai import OpenAI

from .data_contract import MarketViewsReport
from .prompts import (
    FINANCIAL_ANALYST_AGENT_SYSTEM_PROMPT,
    FINANCIAL_ANALYST_AGENT_USER_PROMPT,
)


class FinancialViewsAgent:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,  # for vLLM/Ollama
        temperature: float = 0.1,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def analyze_market_and_generate_views(
        self,
        as_of_date: str,
        asset_universe: Sequence[str] | pd.Index,
        news_payload: dict[str, list[str]] | str,
    ) -> MarketViewsReport:
        assets_list = list(asset_universe)
        formatted_sys_prompt = FINANCIAL_ANALYST_AGENT_SYSTEM_PROMPT.format(
            assets_list=", ".join(assets_list)
        )

        # Build input params
        if isinstance(news_payload, dict):
            context_str = json.dumps(news_payload, indent=2, ensure_ascii=False)
        else:
            context_str = str(news_payload)

        user_prompt = FINANCIAL_ANALYST_AGENT_USER_PROMPT.format(
            as_of_date=as_of_date,
            assets_list=", ".join(assets_list),
            context_str=context_str,
        )

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": formatted_sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=MarketViewsReport,
                temperature=self.temperature,
            )
            return response.choices[0].message.parsed

        except Exception as e:
            # Fallback in case of erros during parsing
            print(f"[Agent Warning] Failed to generate views as of {as_of_date}: {e}")
            return MarketViewsReport(
                as_of_date=as_of_date,
                market_sentiment_summary="Agent fallback: empty views generated due to an error.",
                views=[],
            )

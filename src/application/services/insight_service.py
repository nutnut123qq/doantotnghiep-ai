"""Insight service for generating trading insights."""
import json
import re
from typing import Dict, Any, Optional, List
from src.domain.interfaces.llm_provider import LLMProvider
from src.application.services.prompt_builder import PromptBuilder
from src.shared.utils import normalize_insight_type
from src.shared.logging import get_logger

logger = get_logger(__name__)


class InsightService:
    """Service for generating AI-based trading insights."""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize insight service.
        
        Args:
            llm_provider: LLM provider for generating insights
        """
        self.llm_provider = llm_provider
        logger.info("Initialized InsightService")

    async def generate_insight(
        self,
        symbol: str,
        technical_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate AI-based trading insight (Buy/Sell/Hold signal) for a stock.

        Args:
            symbol: Stock symbol (e.g., VIC, VNM)
            technical_data: Technical indicators (MA, RSI, MACD, etc.)
            fundamental_data: Financial metrics (ROE, ROA, EPS, etc.)
            sentiment_data: News sentiment analysis

        Returns:
            Insight with type (Buy/Sell/Hold), confidence, reasoning, and targets
        """
        logger.info(f"Generating insight for {symbol}")
        
        # Build comprehensive prompt
        prompt = PromptBuilder.build_insight_prompt(
            symbol=symbol,
            technical_data=technical_data,
            fundamental_data=fundamental_data,
            sentiment_data=sentiment_data
        )

        # Generate insight using LLM provider
        try:
            response = await self.llm_provider.generate(prompt)
            logger.debug(f"Received insight response for {symbol}")
        except Exception as e:
            logger.error(f"Error generating insight for {symbol}: {str(e)}")
            raise

        # Parse and structure the response
        insight = self._parse_insight_response(response, symbol)
        logger.info(f"Successfully generated insight for {symbol}: {insight.get('type')} with {insight.get('confidence')}% confidence")

        return insight

    def _parse_insight_response(self, response: str, symbol: str) -> Dict[str, Any]:
        """
        Parse AI response and extract insight data.
        """
        logger.debug(f"Parsing insight response for {symbol}")
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                insight_data = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                insight_data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from insight response for {symbol}: {str(e)}")
            # If JSON parsing fails, create a default response
            insight_data = {
                "type": "Hold",
                "title": "Không thể phân tích",
                "description": "Không đủ dữ liệu để đưa ra khuyến nghị",
                "confidence": 50,
                "reasoning": ["Dữ liệu không đầy đủ"],
                "target_price": None,
                "stop_loss": None
            }

        # Normalize type to match enum values
        insight_type = insight_data.get("type", "Hold")
        insight_type = normalize_insight_type(insight_type)

        # Ensure confidence is between 0-100
        confidence = insight_data.get("confidence", 50)
        if not isinstance(confidence, (int, float)):
            confidence = 50
        confidence = max(0, min(100, int(confidence)))

        # Ensure reasoning is a list
        reasoning = insight_data.get("reasoning", [])
        if not isinstance(reasoning, list):
            reasoning = [str(reasoning)] if reasoning else ["Không có lý do cụ thể"]

        return {
            "symbol": symbol,
            "type": insight_type,
            "title": insight_data.get("title", "Khuyến nghị giao dịch"),
            "description": insight_data.get("description", "Phân tích dựa trên dữ liệu hiện có"),
            "confidence": confidence,
            "reasoning": reasoning,
            "target_price": insight_data.get("target_price"),
            "stop_loss": insight_data.get("stop_loss")
        }

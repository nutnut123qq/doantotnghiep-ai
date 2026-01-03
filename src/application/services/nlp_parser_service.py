from src.infrastructure.llm.gemini_client import GeminiClient
import re


class NLPParserService:
    def __init__(self):
        self.gemini_client = GeminiClient()

    async def parse_alert_intent(self, natural_language_input: str) -> dict:
        prompt = f"""Parse the following alert request and extract structured information:

"{natural_language_input}"

Extract:
1. Stock ticker symbol
2. Condition (price/volume/technical indicator)
3. Threshold value
4. Timeframe
5. Alert type

Respond in JSON format with keys: ticker, condition, threshold, timeframe, alert_type"""

        response = await self.gemini_client.generate(prompt)
        
        # Simple parsing (in production, use structured output from Gemini)
        ticker_match = re.search(r'\b([A-Z]{2,5})\b', natural_language_input.upper())
        ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"
        
        threshold_match = re.search(r'(\d+(?:\.\d+)?)\s*%', natural_language_input)
        threshold = float(threshold_match.group(1)) if threshold_match else 5.0
        
        condition = "price"
        if "volume" in natural_language_input.lower():
            condition = "volume"
        elif "rsi" in natural_language_input.lower() or "macd" in natural_language_input.lower():
            condition = "technical_indicator"
        
        timeframe = "this week"
        if "today" in natural_language_input.lower():
            timeframe = "today"
        elif "month" in natural_language_input.lower():
            timeframe = "this month"
        
        alert_type = "price" if condition == "price" else "volume" if condition == "volume" else "technical_indicator"
        
        return {
            "ticker": ticker,
            "condition": condition,
            "threshold": threshold,
            "timeframe": timeframe,
            "alert_type": alert_type
        }


from src.infrastructure.llm.gemini_client import GeminiClient


class SentimentService:
    def __init__(self):
        self.gemini_client = GeminiClient()

    async def analyze_event(self, event_description: str) -> dict:
        prompt = f"""Analyze the following corporate event and assess its impact:

{event_description}

Please provide:
1. Detailed analysis of the event
2. Expected impact on stock price (positive/negative/neutral)
"""

        response = await self.gemini_client.generate(prompt)
        
        impact = "neutral"
        if "positive" in response.lower() or "increase" in response.lower():
            impact = "positive"
        elif "negative" in response.lower() or "decrease" in response.lower():
            impact = "negative"
        
        return {
            "analysis": response,
            "impact": impact
        }


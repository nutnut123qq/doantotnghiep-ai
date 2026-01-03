from src.application.services.sentiment_service import SentimentService


class AnalyzeEventUseCase:
    def __init__(self):
        self.sentiment_service = SentimentService()

    async def execute(self, event_description: str) -> dict:
        result = await self.sentiment_service.analyze_event(event_description)
        return result


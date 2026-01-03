from src.application.services.summarization_service import SummarizationService


class SummarizeNewsUseCase:
    def __init__(self):
        self.summarization_service = SummarizationService()

    async def execute(self, news_content: str) -> dict:
        result = await self.summarization_service.summarize(news_content)
        return result


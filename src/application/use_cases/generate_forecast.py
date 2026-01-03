from src.application.services.forecast_service import ForecastService


class GenerateForecastUseCase:
    def __init__(self):
        self.forecast_service = ForecastService()

    async def execute(
        self,
        ticker_id: str,
        technical_data: dict = None,
        fundamental_data: dict = None,
        sentiment_data: dict = None
    ) -> dict:
        result = await self.forecast_service.generate_forecast(
            ticker_id=ticker_id,
            technical_data=technical_data,
            fundamental_data=fundamental_data,
            sentiment_data=sentiment_data
        )
        return result


from src.application.services.nlp_parser_service import NLPParserService


class ParseAlertUseCase:
    def __init__(self):
        self.nlp_parser_service = NLPParserService()

    async def execute(self, natural_language_input: str) -> dict:
        result = await self.nlp_parser_service.parse_alert_intent(natural_language_input)
        return result


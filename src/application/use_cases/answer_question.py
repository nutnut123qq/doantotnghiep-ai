from src.application.services.qa_service import QAService


class AnswerQuestionUseCase:
    def __init__(self):
        self.qa_service = QAService()

    async def execute(self, question: str, context: str) -> dict:
        result = await self.qa_service.answer_question(question, context)
        return result


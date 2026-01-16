"""QA API routes."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.application.use_cases.answer_question import AnswerQuestionUseCase
from src.api.dependencies import get_answer_question_use_case

router = APIRouter()


class QARequest(BaseModel):
    """Request model for question answering."""
    question: str
    context: str


class QAResponse(BaseModel):
    """Response model for answer."""
    answer: str
    sources: list[str]


@router.post("/qa", response_model=QAResponse)
async def answer_question(
    request: QARequest,
    use_case: AnswerQuestionUseCase = Depends(get_answer_question_use_case)
):
    """
    Answer a question using RAG.

    Args:
        request: QA request with question and context
        use_case: Answer question use case instance

    Returns:
        Answer with sources
    """
    result = await use_case.execute(request.question, request.context)
    return QAResponse(
        answer=result["answer"],
        sources=result["sources"]
    )

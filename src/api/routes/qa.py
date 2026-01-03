from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.application.use_cases.answer_question import AnswerQuestionUseCase

router = APIRouter()


class QARequest(BaseModel):
    question: str
    context: str


class QAResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/qa", response_model=QAResponse)
async def answer_question(request: QARequest):
    try:
        use_case = AnswerQuestionUseCase()
        result = await use_case.execute(request.question, request.context)
        return QAResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


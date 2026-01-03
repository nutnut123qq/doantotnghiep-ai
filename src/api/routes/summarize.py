from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.application.use_cases.summarize_news import SummarizeNewsUseCase

router = APIRouter()


class SummarizeRequest(BaseModel):
    content: str


class SummarizeResponse(BaseModel):
    summary: str
    sentiment: str
    impact_assessment: str


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_news(request: SummarizeRequest):
    try:
        use_case = SummarizeNewsUseCase()
        result = await use_case.execute(request.content)
        return SummarizeResponse(
            summary=result["summary"],
            sentiment=result["sentiment"],
            impact_assessment=result["impact_assessment"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


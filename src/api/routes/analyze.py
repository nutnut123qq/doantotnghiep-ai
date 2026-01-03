from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.application.use_cases.analyze_event import AnalyzeEventUseCase

router = APIRouter()


class AnalyzeRequest(BaseModel):
    description: str


class AnalyzeResponse(BaseModel):
    analysis: str
    impact: str


@router.post("/analyze-event", response_model=AnalyzeResponse)
async def analyze_event(request: AnalyzeRequest):
    try:
        use_case = AnalyzeEventUseCase()
        result = await use_case.execute(request.description)
        return AnalyzeResponse(
            analysis=result["analysis"],
            impact=result["impact"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.application.use_cases.parse_alert import ParseAlertUseCase

router = APIRouter()


class ParseAlertRequest(BaseModel):
    input: str


class ParseAlertResponse(BaseModel):
    ticker: str
    condition: str
    threshold: float
    timeframe: str
    alert_type: str


@router.post("/parse-alert", response_model=ParseAlertResponse)
async def parse_alert(request: ParseAlertRequest):
    try:
        use_case = ParseAlertUseCase()
        result = await use_case.execute(request.input)
        return ParseAlertResponse(
            ticker=result["ticker"],
            condition=result["condition"],
            threshold=result["threshold"],
            timeframe=result["timeframe"],
            alert_type=result["alert_type"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


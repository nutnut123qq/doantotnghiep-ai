"""
Stock Data API Routes
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta

from ...application.services.stock_data_service import StockDataService

router = APIRouter(prefix="/api/stock", tags=["stock"])
stock_service = StockDataService()


class StockQuoteResponse(BaseModel):
    symbol: str
    currentPrice: float
    previousClose: float
    change: float
    changePercent: float
    volume: int
    high: float
    low: float
    open: float
    lastUpdated: str


class SymbolInfo(BaseModel):
    symbol: str
    name: Optional[str] = None
    exchange: Optional[str] = None
    industry: Optional[str] = None


class HistoricalDataPoint(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class SymbolsResponse(BaseModel):
    symbols: List[SymbolInfo]


@router.get("/symbols", response_model=SymbolsResponse)
async def get_all_symbols(
    exchange: Optional[str] = Query(None, description="Sàn giao dịch: HOSE, HNX, UPCOM")
):
    """
    Lấy danh sách tất cả mã chứng khoán
    """
    try:
        symbols_data = stock_service.get_all_symbols(exchange)
        symbols = [
            SymbolInfo(
                symbol=s.get('ticker', s.get('symbol', '')),
                name=s.get('organ_name', s.get('company_name', s.get('name'))),
                exchange=s.get('exchange', ''),
                industry=s.get('icb_name3', s.get('industry'))
            )
            for s in symbols_data
        ]
        return SymbolsResponse(symbols=symbols)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quote/{symbol}", response_model=StockQuoteResponse)
async def get_stock_quote(
    symbol: str,
    source: str = Query("VCI", description="Nguồn dữ liệu: VCI, TCBS")
):
    """
    Lấy giá hiện tại của một mã chứng khoán
    """
    try:
        quote = stock_service.get_stock_quote(symbol, source)
        if not quote:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu cho mã {symbol}")
        return quote
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quotes", response_model=List[StockQuoteResponse])
async def get_multiple_quotes(
    symbols: List[str],
    source: str = Query("VCI", description="Nguồn dữ liệu: VCI, TCBS")
):
    """
    Lấy giá của nhiều mã chứng khoán cùng lúc
    """
    try:
        quotes = stock_service.get_multiple_quotes(symbols, source)
        return quotes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{symbol}", response_model=List[HistoricalDataPoint])
async def get_historical_data(
    symbol: str,
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc (YYYY-MM-DD)"),
    interval: str = Query("1D", description="Khoảng thời gian: 1D, 1W, 1M"),
    source: str = Query("VCI", description="Nguồn dữ liệu: VCI, TCBS")
):
    """
    Lấy dữ liệu lịch sử của mã chứng khoán
    """
    try:
        # Nếu không có start_date, lấy 30 ngày gần nhất
        if not start_date:
            start = datetime.now() - timedelta(days=30)
            start_date = start.strftime('%Y-%m-%d')
        
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        data = stock_service.get_historical_data(symbol, start_date, end_date, interval, source)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vn30")
async def get_vn30_quotes(source: str = Query("VCI", description="Nguồn dữ liệu: VCI, TCBS")):
    """
    Lấy giá của các mã VN30
    """
    try:
        # Danh sách mã VN30 (có thể cập nhật định kỳ)
        vn30_symbols = [
            'VIC', 'VNM', 'VCB', 'VRE', 'VHM', 'GAS', 'MSN', 'BID', 'CTG', 'HPG',
            'TCB', 'MBB', 'VPB', 'PLX', 'SAB', 'VJC', 'GVR', 'FPT', 'POW', 'SSI',
            'MWG', 'HDB', 'ACB', 'TPB', 'STB', 'PDR', 'VIB', 'BCM', 'KDH', 'NVL'
        ]
        
        quotes = stock_service.get_multiple_quotes(vn30_symbols, source)
        return quotes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


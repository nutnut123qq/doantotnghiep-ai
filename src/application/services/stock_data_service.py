"""
Stock Data Service - Lấy dữ liệu chứng khoán từ vnstock
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from vnstock import Vnstock, Listing
import logging

logger = logging.getLogger(__name__)


class StockDataService:
    """Service để lấy dữ liệu chứng khoán từ vnstock"""
    
    def __init__(self):
        self.listing = Listing()
        self._cache = {}
        self._cache_ttl = 60  # Cache 60 giây cho real-time data
    
    def get_all_symbols(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả mã chứng khoán
        
        Args:
            exchange: Sàn giao dịch (HOSE, HNX, UPCOM)
        
        Returns:
            List of stock symbols with basic info
        """
        try:
            df = self.listing.all_symbols()
            
            if exchange:
                df = df[df['exchange'] == exchange.upper()]
            
            # Convert DataFrame to list of dicts
            symbols = df.to_dict('records')
            
            return symbols
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            return []
    
    def get_stock_quote(self, symbol: str, source: str = 'VCI') -> Dict[str, Any]:
        """
        Lấy giá hiện tại của mã chứng khoán
        
        Args:
            symbol: Mã chứng khoán (VD: VIC, VNM)
            source: Nguồn dữ liệu (VCI, TCBS)
        
        Returns:
            Dict chứa thông tin giá hiện tại
        """
        try:
            stock = Vnstock().stock(symbol=symbol.upper(), source=source)
            
            # Lấy dữ liệu 2 ngày gần nhất để tính change
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            df = stock.quote.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1D'
            )
            
            if df.empty:
                return {}
            
            # Lấy dòng cuối cùng (ngày gần nhất)
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            return {
                'symbol': symbol.upper(),
                'currentPrice': float(latest['close']),
                'previousClose': float(previous['close']),
                'change': float(latest['close'] - previous['close']),
                'changePercent': float((latest['close'] - previous['close']) / previous['close'] * 100),
                'volume': int(latest['volume']) if 'volume' in latest else 0,
                'high': float(latest['high']),
                'low': float(latest['low']),
                'open': float(latest['open']),
                'lastUpdated': latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
            }
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {str(e)}")
            return {}
    
    def get_multiple_quotes(self, symbols: List[str], source: str = 'VCI') -> List[Dict[str, Any]]:
        """
        Lấy giá của nhiều mã chứng khoán
        
        Args:
            symbols: Danh sách mã chứng khoán
            source: Nguồn dữ liệu
        
        Returns:
            List of stock quotes
        """
        quotes = []
        for symbol in symbols:
            quote = self.get_stock_quote(symbol, source)
            if quote:
                quotes.append(quote)
        
        return quotes
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1D',
        source: str = 'VCI'
    ) -> List[Dict[str, Any]]:
        """
        Lấy dữ liệu lịch sử
        
        Args:
            symbol: Mã chứng khoán
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (YYYY-MM-DD)
            interval: Khoảng thời gian (1D, 1W, 1M)
            source: Nguồn dữ liệu
        
        Returns:
            List of historical data points
        """
        try:
            stock = Vnstock().stock(symbol=symbol.upper(), source=source)
            df = stock.quote.history(start=start_date, end=end_date, interval=interval)
            
            # Convert DataFrame to list of dicts
            data = df.reset_index().to_dict('records')
            
            # Convert datetime to string
            for item in data:
                if 'time' in item and hasattr(item['time'], 'isoformat'):
                    item['time'] = item['time'].isoformat()
            
            return data
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return []


"""Prompt builder utilities for AI services."""
from typing import Optional, Dict, Any
from src.shared.constants import TIME_HORIZON_MAP
from src.shared.utils import (
    format_technical_data,
    format_fundamental_data,
    format_sentiment_data
)
from src.shared.logging import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Utility class for building AI prompts."""
    
    @staticmethod
    def build_forecast_prompt(
        symbol: str,
        technical_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        time_horizon: str = "short"
    ) -> str:
        """
        Build prompt for stock forecast.
        
        Args:
            symbol: Stock symbol
            technical_data: Technical indicators
            fundamental_data: Fundamental metrics
            sentiment_data: Sentiment analysis
            time_horizon: Forecast time period (short, medium, long)
            
        Returns:
            Formatted prompt string
        """
        time_period = TIME_HORIZON_MAP.get(time_horizon, "1-5 ngày tới")
        
        prompt = f"""Bạn là chuyên gia phân tích chứng khoán Việt Nam. Hãy dự báo xu hướng cổ phiếu {symbol} trong {time_period}.

DỮ LIỆU PHÂN TÍCH:

"""
        
        # Add technical analysis
        prompt += format_technical_data(technical_data)
        
        # Add fundamental analysis
        prompt += format_fundamental_data(fundamental_data)
        
        # Add sentiment analysis
        prompt += format_sentiment_data(sentiment_data)
        
        prompt += """HÃY CUNG CẤP DỰ BÁO CHI TIẾT:

1. **Xu hướng dự báo**: Tăng/Giảm/Đi ngang
2. **Mức độ tin cậy**: Cao (>70%) / Trung bình (50-70%) / Thấp (<50%)
3. **Mục tiêu giá**:
   - Giá mục tiêu (target price)
   - Giá hỗ trợ (support level)
   - Giá kháng cự (resistance level)
4. **Yếu tố chính** (2-3 yếu tố quan trọng nhất)
5. **Rủi ro** (2-3 rủi ro cần lưu ý)
6. **Khuyến nghị**: Mua/Giữ/Bán

Trả lời bằng tiếng Việt, có cấu trúc rõ ràng và dễ hiểu."""
        
        return prompt
    
    @staticmethod
    def build_insight_prompt(
        symbol: str,
        technical_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for trading insight.
        
        Args:
            symbol: Stock symbol
            technical_data: Technical indicators
            fundamental_data: Fundamental metrics
            sentiment_data: Sentiment analysis
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Bạn là chuyên gia phân tích chứng khoán Việt Nam. Hãy đưa ra khuyến nghị giao dịch (MUA/BÁN/GIỮ) cho cổ phiếu {symbol} dựa trên dữ liệu phân tích.

DỮ LIỆU PHÂN TÍCH:

"""
        
        # Add technical analysis
        if technical_data:
            prompt += f"""1. CHỈ SỐ KỸ THUẬT:
- MA (Moving Average): {technical_data.get('ma', 'N/A')}
- RSI (Relative Strength Index): {technical_data.get('rsi', 'N/A')}
- MACD: {technical_data.get('macd', 'N/A')}
- Volume: {technical_data.get('volume', 'N/A')}
- Price Trend: {technical_data.get('trend', 'N/A')}

"""
        
        # Add fundamental analysis
        if fundamental_data:
            prompt += f"""2. CHỈ SỐ TÀI CHÍNH:
- ROE (Return on Equity): {fundamental_data.get('roe', 'N/A')}%
- ROA (Return on Assets): {fundamental_data.get('roa', 'N/A')}%
- EPS (Earnings Per Share): {fundamental_data.get('eps', 'N/A')}
- P/E Ratio: {fundamental_data.get('pe', 'N/A')}
- Revenue Growth: {fundamental_data.get('revenue_growth', 'N/A')}%

"""
        
        # Add sentiment analysis
        if sentiment_data:
            prompt += f"""3. TÂM LÝ THỊ TRƯỜNG:
- Sentiment Score: {sentiment_data.get('score', 'N/A')}
- Overall Sentiment: {sentiment_data.get('sentiment', 'N/A')}
- Recent News: {sentiment_data.get('recent_news', 'N/A')}

"""
        
        prompt += """YÊU CẦU:
Hãy phân tích và đưa ra khuyến nghị giao dịch với format JSON sau:

{
  "type": "Buy" hoặc "Sell" hoặc "Hold",
  "title": "Tiêu đề ngắn gọn của insight (ví dụ: 'Strong Buy Signal Detected')",
  "description": "Mô tả ngắn gọn về tín hiệu (1-2 câu)",
  "confidence": 0-100 (điểm tin cậy),
  "reasoning": ["Lý do 1", "Lý do 2", "Lý do 3"] (danh sách các yếu tố chính),
  "target_price": giá mục tiêu nếu là Buy (optional, có thể null),
  "stop_loss": giá cắt lỗ nếu là Buy hoặc Sell (optional, có thể null)
}

Lưu ý:
- "Buy": Cổ phiếu có tiềm năng tăng giá mạnh
- "Sell": Cổ phiếu có nguy cơ giảm giá hoặc nên chốt lời
- "Hold": Cổ phiếu ổn định, không có tín hiệu rõ ràng
- Confidence: 0-100, càng cao càng tin cậy
- Reasoning: Liệt kê 3-5 yếu tố chính ảnh hưởng đến quyết định
"""
        
        return prompt

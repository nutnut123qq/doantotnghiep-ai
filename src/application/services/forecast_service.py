import json
import re
from typing import Dict, Any, Optional
from src.infrastructure.llm.gemini_client import GeminiClient


class ForecastService:
    def __init__(self):
        self.gemini_client = GeminiClient()

    async def generate_forecast(
        self,
        symbol: str,
        technical_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        time_horizon: str = "short"  # short (1-5 days), medium (1-4 weeks), long (1-3 months)
    ) -> Dict[str, Any]:
        """
        Generate AI-based stock forecast using multiple data sources

        Args:
            symbol: Stock symbol (e.g., VIC, VNM)
            technical_data: Technical indicators (MA, RSI, MACD, etc.)
            fundamental_data: Financial metrics (ROE, ROA, EPS, etc.)
            sentiment_data: News sentiment analysis
            time_horizon: Forecast time period

        Returns:
            Forecast with trend, confidence, price targets, and analysis
        """

        # Build comprehensive prompt
        prompt = self._build_forecast_prompt(
            symbol, technical_data, fundamental_data, sentiment_data, time_horizon
        )

        # Generate forecast using Gemini
        response = await self.gemini_client.generate(prompt)

        # Parse and structure the response
        forecast = self._parse_forecast_response(response, symbol, time_horizon)

        return forecast

    def _build_forecast_prompt(
        self,
        symbol: str,
        technical_data: Optional[Dict[str, Any]],
        fundamental_data: Optional[Dict[str, Any]],
        sentiment_data: Optional[Dict[str, Any]],
        time_horizon: str
    ) -> str:
        """Build detailed prompt for AI forecast"""

        time_period_map = {
            "short": "1-5 ngày tới",
            "medium": "1-4 tuần tới",
            "long": "1-3 tháng tới"
        }

        prompt = f"""Bạn là chuyên gia phân tích chứng khoán Việt Nam. Hãy dự báo xu hướng cổ phiếu {symbol} trong {time_period_map.get(time_horizon, '1-5 ngày tới')}.

DỮ LIỆU PHÂN TÍCH:

"""

        # Add technical analysis
        if technical_data:
            prompt += f"""1. CHỈ SỐ KỸ THUẬT:
- MA (Moving Average): {technical_data.get('ma', 'N/A')}
- RSI (Relative Strength Index): {technical_data.get('rsi', 'N/A')}
- MACD: {technical_data.get('macd', 'N/A')}
- Bollinger Bands: {technical_data.get('bollinger', 'N/A')}
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
- Profit Margin: {fundamental_data.get('profit_margin', 'N/A')}%

"""

        # Add sentiment analysis
        if sentiment_data:
            prompt += f"""3. PHÂN TÍCH TÂM LÝ THỊ TRƯỜNG:
- Sentiment Score: {sentiment_data.get('score', 'N/A')}
- News Sentiment: {sentiment_data.get('sentiment', 'N/A')}
- Social Media Buzz: {sentiment_data.get('social_buzz', 'N/A')}
- Recent News: {sentiment_data.get('recent_news', 'N/A')}

"""

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

    def _parse_forecast_response(
        self,
        response: str,
        symbol: str,
        time_horizon: str
    ) -> Dict[str, Any]:
        """Parse AI response into structured forecast"""

        # Extract trend
        trend = "Sideways"
        if any(word in response.lower() for word in ["tăng", "up", "bullish", "tích cực"]):
            trend = "Up"
        elif any(word in response.lower() for word in ["giảm", "down", "bearish", "tiêu cực"]):
            trend = "Down"

        # Extract confidence level
        confidence = "Medium"
        confidence_score = 50.0

        if any(word in response.lower() for word in ["cao", "high", ">70", "mạnh"]):
            confidence = "High"
            confidence_score = 75.0
        elif any(word in response.lower() for word in ["thấp", "low", "<50", "yếu"]):
            confidence = "Low"
            confidence_score = 35.0

        # Extract recommendation
        recommendation = "Hold"
        if any(word in response.lower() for word in ["mua", "buy", "tích lũy"]):
            recommendation = "Buy"
        elif any(word in response.lower() for word in ["bán", "sell", "thoát"]):
            recommendation = "Sell"

        # Extract key drivers and risks
        key_drivers = self._extract_list_items(response, ["yếu tố", "driver", "lý do"])
        risks = self._extract_list_items(response, ["rủi ro", "risk", "nguy cơ"])

        return {
            "symbol": symbol,
            "trend": trend,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "time_horizon": time_horizon,
            "recommendation": recommendation,
            "key_drivers": key_drivers[:3] if key_drivers else ["Phân tích kỹ thuật", "Tâm lý thị trường"],
            "risks": risks[:3] if risks else ["Biến động thị trường", "Rủi ro vĩ mô"],
            "analysis": response,
            "generated_at": None  # Will be set by API
        }

    def _extract_list_items(self, text: str, keywords: list) -> list:
        """Extract list items from text based on keywords"""
        items = []
        lines = text.split('\n')

        in_section = False
        for line in lines:
            # Check if we're in the relevant section
            if any(keyword in line.lower() for keyword in keywords):
                in_section = True
                continue

            # Extract bullet points or numbered items
            if in_section:
                # Stop at next section
                if line.strip().startswith('#') or (line.strip() and line.strip()[0].isdigit() and '.' in line[:3]):
                    if not any(keyword in line.lower() for keyword in keywords):
                        in_section = False
                        continue

                # Extract item
                match = re.match(r'^[\s\-\*\d\.]+(.+)$', line)
                if match:
                    item = match.group(1).strip()
                    if item and len(item) > 5:
                        items.append(item)

        return items


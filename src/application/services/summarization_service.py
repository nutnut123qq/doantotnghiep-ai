import json
import re
from src.infrastructure.llm.gemini_client import GeminiClient


class SummarizationService:
    def __init__(self):
        self.gemini_client = GeminiClient()

    async def summarize(self, content: str) -> dict:
        prompt = f"""Bạn là chuyên gia phân tích tài chính. Hãy phân tích bài viết tin tức sau về thị trường chứng khoán Việt Nam:

{content}

Hãy trả lời theo định dạng JSON sau:
{{
    "summary": "Tóm tắt ngắn gọn 2-3 câu về nội dung chính",
    "sentiment": "positive/negative/neutral",
    "impact_assessment": "Đánh giá tác động đến thị trường/cổ phiếu liên quan",
    "key_points": ["Điểm chính 1", "Điểm chính 2", "Điểm chính 3"]
}}

Lưu ý:
- Sentiment: positive (tích cực), negative (tiêu cực), neutral (trung lập)
- Impact assessment: Phân tích cụ thể tác động đến giá cổ phiếu, xu hướng thị trường
- Key points: Các điểm quan trọng nhất trong bài viết
"""

        response = await self.gemini_client.generate(prompt)

        try:
            # Try to parse JSON response
            # Remove markdown code blocks if present
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()

            result = json.loads(json_str)

            # Validate required fields
            if "summary" not in result:
                result["summary"] = "Không thể tạo tóm tắt"
            if "sentiment" not in result:
                result["sentiment"] = "neutral"
            if "impact_assessment" not in result:
                result["impact_assessment"] = "Chưa có đánh giá tác động"
            if "key_points" not in result:
                result["key_points"] = []

            # Normalize sentiment value
            sentiment_lower = result["sentiment"].lower()
            if "positive" in sentiment_lower or "tích cực" in sentiment_lower:
                result["sentiment"] = "positive"
            elif "negative" in sentiment_lower or "tiêu cực" in sentiment_lower:
                result["sentiment"] = "negative"
            else:
                result["sentiment"] = "neutral"

            return result

        except json.JSONDecodeError:
            # Fallback to simple parsing if JSON parsing fails
            summary = self._extract_summary(response)
            sentiment = self._extract_sentiment(response)
            impact = self._extract_impact(response)

            return {
                "summary": summary,
                "sentiment": sentiment,
                "impact_assessment": impact,
                "key_points": []
            }

    def _extract_summary(self, text: str) -> str:
        """Extract summary from unstructured text"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            # Return first non-empty line as summary
            return lines[0][:500]  # Limit to 500 chars
        return "Không thể tạo tóm tắt"

    def _extract_sentiment(self, text: str) -> str:
        """Extract sentiment from unstructured text"""
        text_lower = text.lower()

        positive_keywords = ["positive", "tích cực", "tăng", "tốt", "khả quan", "lạc quan"]
        negative_keywords = ["negative", "tiêu cực", "giảm", "xấu", "bi quan", "lo ngại"]

        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _extract_impact(self, text: str) -> str:
        """Extract impact assessment from unstructured text"""
        # Look for impact-related sentences
        sentences = text.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ["impact", "tác động", "ảnh hưởng", "affect"]):
                return sentence.strip()

        return "Chưa có đánh giá tác động cụ thể"


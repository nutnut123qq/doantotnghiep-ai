from src.infrastructure.llm.gemini_client import GeminiClient
from src.infrastructure.vector_store.embedding_service import EmbeddingService
from src.infrastructure.vector_store.qdrant_client import QdrantClient


class QAService:
    def __init__(self):
        self.gemini_client = GeminiClient()
        self.embedding_service = EmbeddingService()
        self.qdrant_client = QdrantClient()

    async def answer_question(self, question: str, context: str) -> dict:
        # Generate embedding for the question
        question_embedding = await self.embedding_service.generate_embedding(question)

        # Search for relevant context in vector database
        relevant_docs = await self.qdrant_client.search(question_embedding, limit=3)

        # Combine context with retrieved documents
        full_context = context
        if relevant_docs:
            full_context += "\n\nThông tin liên quan:\n" + "\n".join([doc.get("text", "") for doc in relevant_docs])

        # Enhanced prompt for financial reports with Vietnamese support
        prompt = f"""Bạn là một chuyên gia phân tích tài chính. Dựa trên dữ liệu báo cáo tài chính được cung cấp, hãy trả lời câu hỏi một cách chi tiết và chính xác.

Hướng dẫn:
- Trả lời bằng tiếng Việt
- Trích dẫn số liệu cụ thể từ báo cáo
- Giải thích các chỉ số tài chính nếu cần
- Đưa ra phân tích và đánh giá nếu phù hợp
- Nếu không có đủ thông tin, hãy nói rõ

Dữ liệu báo cáo tài chính:
{full_context}

Câu hỏi: {question}

Trả lời:"""

        answer = await self.gemini_client.generate(prompt)

        sources = [doc.get("source", "") for doc in relevant_docs if doc.get("source")]

        return {
            "answer": answer,
            "sources": sources
        }

    async def analyze_financial_metrics(self, financial_data: dict) -> dict:
        """
        Analyze financial metrics and provide insights
        """
        prompt = f"""Phân tích các chỉ số tài chính sau và đưa ra đánh giá:

Dữ liệu tài chính:
{financial_data}

Hãy phân tích:
1. Tình hình tài chính tổng quan
2. Các chỉ số quan trọng (ROE, ROA, EPS, etc.)
3. Xu hướng tăng trưởng
4. Điểm mạnh và điểm yếu
5. Khuyến nghị cho nhà đầu tư

Trả lời bằng tiếng Việt, có cấu trúc rõ ràng."""

        analysis = await self.gemini_client.generate(prompt)

        return {
            "analysis": analysis,
            "metrics": financial_data
        }


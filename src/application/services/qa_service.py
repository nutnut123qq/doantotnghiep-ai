"""QA service for answering questions with RAG."""
from typing import Dict, Any
from src.domain.interfaces.llm_provider import LLMProvider
from src.domain.interfaces.vector_store import VectorStore
from src.domain.interfaces.embedding_provider import EmbeddingProvider
from src.shared.logging import get_logger

logger = get_logger(__name__)


class QAService:
    """Service for answering questions using RAG (Retrieval-Augmented Generation)."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        vector_store: VectorStore,
        embedding_service: EmbeddingProvider
    ):
        """
        Initialize QA service.
        
        Args:
            llm_provider: LLM provider for generating answers
            vector_store: Vector store for document retrieval
            embedding_service: Service for generating embeddings
        """
        self.llm_provider = llm_provider
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        logger.info("Initialized QAService")

    async def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """
        Answer a question using RAG with context.
        
        Args:
            question: The question to answer
            context: Additional context to include
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Answering question: {question[:100]}...")
        
        try:
            # Generate embedding for the question
            question_embedding = await self.embedding_service.generate_embedding(question)
            logger.debug("Generated question embedding")

            # Search for relevant context in vector database
            relevant_docs = await self.vector_store.search(question_embedding, limit=3)
            logger.debug(f"Found {len(relevant_docs)} relevant documents")

            # Combine context with retrieved documents
            full_context = context
            if relevant_docs:
                full_context += "\n\nThông tin liên quan:\n" + "\n".join([
                    doc.get("text", "") for doc in relevant_docs
                ])

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

            answer = await self.llm_provider.generate(prompt)
            logger.info("Successfully generated answer")

            sources = [
                doc.get("source", "") for doc in relevant_docs if doc.get("source")
            ]

            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise

    async def analyze_financial_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze financial metrics and provide insights.
        
        Args:
            financial_data: Financial data dictionary
            
        Returns:
            Dictionary with analysis and metrics
        """
        logger.info("Analyzing financial metrics")
        
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

        try:
            analysis = await self.llm_provider.generate(prompt)
            logger.info("Successfully analyzed financial metrics")

            return {
                "analysis": analysis,
                "metrics": financial_data
            }
        except Exception as e:
            logger.error(f"Error analyzing financial metrics: {str(e)}")
            raise

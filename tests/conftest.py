"""Pytest configuration and fixtures for AI Service tests."""
import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List
from src.domain.interfaces.llm_provider import LLMProvider
from src.domain.interfaces.vector_store import VectorStore
from src.domain.interfaces.embedding_provider import EmbeddingProvider
from src.api.main import app
from src.application.services.qa_service import QAService
from src.application.services.forecast_service import ForecastService
from src.application.services.insight_service import InsightService
from src.application.services.summarization_service import SummarizationService
from src.application.services.sentiment_service import SentimentService
from src.application.services.nlp_parser_service import NLPParserService
from src.application.services.stock_data_service import StockDataService
from src.application.use_cases.answer_question import AnswerQuestionUseCase
from src.application.use_cases.generate_forecast import GenerateForecastUseCase
from src.application.use_cases.generate_insight import GenerateInsightUseCase
from src.application.use_cases.summarize_news import SummarizeNewsUseCase
from src.application.use_cases.analyze_event import AnalyzeEventUseCase
from src.application.use_cases.parse_alert import ParseAlertUseCase


@pytest.fixture
def mock_llm_provider() -> Mock:
    """Create a mock LLM provider."""
    mock = Mock(spec=LLMProvider)
    mock.generate = AsyncMock(return_value="Mocked LLM response")
    return mock


@pytest.fixture
def mock_vector_store() -> Mock:
    """Create a mock vector store."""
    mock = Mock(spec=VectorStore)
    mock.search = AsyncMock(return_value=[
        {"text": "Test document", "source": "test.pdf", "score": 0.9}
    ])
    mock.upsert = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_embedding_provider() -> Mock:
    """Create a mock embedding provider."""
    mock = Mock(spec=EmbeddingProvider)
    mock.generate_embedding = AsyncMock(return_value=[0.1] * 384)
    return mock


@pytest.fixture
def sample_forecast_response() -> str:
    """Sample forecast response from LLM."""
    return """Dựa trên phân tích:

Xu hướng dự báo: Tăng
Mức độ tin cậy: Cao (>70%)

Yếu tố chính:
1. Phân tích kỹ thuật cho thấy xu hướng tăng mạnh
2. Tâm lý thị trường tích cực
3. Chỉ số tài chính tốt

Rủi ro:
1. Biến động thị trường
2. Rủi ro vĩ mô

Khuyến nghị: Mua"""


@pytest.fixture
def sample_insight_json() -> str:
    """Sample insight JSON response from LLM."""
    return """{
  "type": "Buy",
  "title": "Strong Buy Signal Detected",
  "description": "Cổ phiếu có tiềm năng tăng giá mạnh",
  "confidence": 85,
  "reasoning": ["Phân tích kỹ thuật tích cực", "Tâm lý thị trường tốt", "Chỉ số tài chính mạnh"],
  "target_price": 125000,
  "stop_loss": 110000
}"""


@pytest.fixture
def override_dependencies(mock_llm_provider, mock_vector_store, mock_embedding_provider):
    """Override FastAPI dependencies with mocks for testing."""
    from src.api import dependencies
    
    # Override infrastructure dependencies
    app.dependency_overrides[dependencies.get_llm_provider] = lambda: mock_llm_provider
    app.dependency_overrides[dependencies.get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[dependencies.get_embedding_service] = lambda: mock_embedding_provider
    
    # Override services that depend on infrastructure
    mock_qa_service = QAService(mock_llm_provider, mock_vector_store, mock_embedding_provider)
    mock_forecast_service = ForecastService(mock_llm_provider)
    mock_insight_service = InsightService(mock_llm_provider)
    mock_summarization_service = SummarizationService(mock_llm_provider)
    mock_sentiment_service = SentimentService(mock_llm_provider)
    mock_nlp_parser_service = NLPParserService(mock_llm_provider)
    mock_stock_data_service = Mock(spec=StockDataService)
    
    app.dependency_overrides[dependencies.get_qa_service] = lambda: mock_qa_service
    app.dependency_overrides[dependencies.get_forecast_service] = lambda: mock_forecast_service
    app.dependency_overrides[dependencies.get_insight_service] = lambda: mock_insight_service
    app.dependency_overrides[dependencies.get_summarization_service] = lambda: mock_summarization_service
    app.dependency_overrides[dependencies.get_sentiment_service] = lambda: mock_sentiment_service
    app.dependency_overrides[dependencies.get_nlp_parser_service] = lambda: mock_nlp_parser_service
    app.dependency_overrides[dependencies.get_stock_data_service] = lambda: mock_stock_data_service
    
    # Override use cases
    app.dependency_overrides[dependencies.get_answer_question_use_case] = lambda: AnswerQuestionUseCase(mock_qa_service)
    app.dependency_overrides[dependencies.get_generate_forecast_use_case] = lambda: GenerateForecastUseCase(mock_forecast_service)
    app.dependency_overrides[dependencies.get_generate_insight_use_case] = lambda: GenerateInsightUseCase(mock_insight_service)
    app.dependency_overrides[dependencies.get_summarize_news_use_case] = lambda: SummarizeNewsUseCase(mock_summarization_service)
    app.dependency_overrides[dependencies.get_analyze_event_use_case] = lambda: AnalyzeEventUseCase(mock_sentiment_service)
    app.dependency_overrides[dependencies.get_parse_alert_use_case] = lambda: ParseAlertUseCase(mock_nlp_parser_service)
    
    yield {
        "llm_provider": mock_llm_provider,
        "vector_store": mock_vector_store,
        "embedding_provider": mock_embedding_provider,
        "qa_service": mock_qa_service,
        "forecast_service": mock_forecast_service,
        "insight_service": mock_insight_service,
        "stock_data_service": mock_stock_data_service,
    }
    
    # Cleanup: clear all overrides
    app.dependency_overrides.clear()

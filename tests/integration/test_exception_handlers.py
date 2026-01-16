"""Integration tests for exception handlers."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock
from src.api.main import app
from src.shared.exceptions import (
    LLMQuotaExceededError,
    LLMProviderError,
    VectorStoreError,
    ValidationError,
    ServiceUnavailableError,
    NotFoundError
)
from src.domain.interfaces.llm_provider import LLMProvider
from src.domain.interfaces.vector_store import VectorStore
from src.domain.interfaces.embedding_provider import EmbeddingProvider
from src.application.services.qa_service import QAService
from src.api.dependencies import get_answer_question_use_case


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.integration
def test_llm_quota_exceeded_handler(client):
    """Test LLMQuotaExceededError handler returns 503."""
    # Create a mock LLM provider that raises LLMQuotaExceededError
    mock_llm = Mock(spec=LLMProvider)
    mock_llm.generate = AsyncMock(side_effect=LLMQuotaExceededError("Quota exceeded"))
    
    mock_vector = Mock(spec=VectorStore)
    mock_vector.search = AsyncMock(return_value=[])
    
    mock_embedding = Mock(spec=EmbeddingProvider)
    mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 384)
    
    # Override dependencies
    app.dependency_overrides[get_answer_question_use_case] = lambda: Mock(
        execute=AsyncMock(side_effect=LLMQuotaExceededError("Quota exceeded"))
    )
    
    try:
        response = client.post(
            "/api/qa",
            json={"question": "test", "context": "test"}
        )
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "LLM quota exceeded"
        assert data["type"] == "LLMQuotaExceededError"
        assert "request_id" in data
        assert "message" in data
    finally:
        app.dependency_overrides.clear()


@pytest.mark.integration
def test_validation_error_handler(client):
    """Test ValidationError handler returns 400."""
    # Override to raise ValidationError
    from src.api.dependencies import get_stock_data_service
    from src.application.services.stock_data_service import StockDataService
    
    mock_service = Mock(spec=StockDataService)
    mock_service.get_stock_quote = Mock(side_effect=ValidationError("Invalid symbol"))
    
    app.dependency_overrides[get_stock_data_service] = lambda: mock_service
    
    try:
        response = client.get("/api/stock/quote/INVALID")
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "Validation error"
        assert data["type"] == "ValidationError"
        assert "request_id" in data
    finally:
        app.dependency_overrides.clear()


@pytest.mark.integration
def test_vector_store_error_handler(client):
    """Test VectorStoreError handler returns 503."""
    # Override to raise VectorStoreError
    app.dependency_overrides[get_answer_question_use_case] = lambda: Mock(
        execute=AsyncMock(side_effect=VectorStoreError("Vector store unavailable"))
    )
    
    try:
        response = client.post(
            "/api/qa",
            json={"question": "test", "context": "test"}
        )
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "Vector store error"
        assert data["type"] == "VectorStoreError"
        assert "request_id" in data
    finally:
        app.dependency_overrides.clear()


@pytest.mark.integration
def test_service_unavailable_error_handler(client):
    """Test ServiceUnavailableError handler returns 503."""
    from src.api.dependencies import get_stock_data_service
    from src.application.services.stock_data_service import StockDataService
    
    mock_service = Mock(spec=StockDataService)
    mock_service.get_all_symbols = Mock(side_effect=ServiceUnavailableError("Service down"))
    
    app.dependency_overrides[get_stock_data_service] = lambda: mock_service
    
    try:
        response = client.get("/api/stock/symbols")
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "Service unavailable"
        assert data["type"] == "ServiceUnavailableError"
        assert "request_id" in data
    finally:
        app.dependency_overrides.clear()


@pytest.mark.integration
def test_not_found_error_handler(client):
    """Test NotFoundError handler returns 404."""
    from src.api.dependencies import get_stock_data_service
    from src.application.services.stock_data_service import StockDataService
    
    mock_service = Mock(spec=StockDataService)
    mock_service.get_stock_quote = Mock(side_effect=NotFoundError("Symbol not found"))
    
    app.dependency_overrides[get_stock_data_service] = lambda: mock_service
    
    try:
        response = client.get("/api/stock/quote/NOTFOUND")
        
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "Resource not found"
        assert data["type"] == "NotFoundError"
        assert "request_id" in data
    finally:
        app.dependency_overrides.clear()


@pytest.mark.integration
def test_general_exception_handler(client):
    """Test general Exception handler returns 500 without leaking details."""
    # Override to raise generic Exception
    app.dependency_overrides[get_answer_question_use_case] = lambda: Mock(
        execute=AsyncMock(side_effect=RuntimeError("Internal error"))
    )
    
    try:
        response = client.post(
            "/api/qa",
            json={"question": "test", "context": "test"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Internal server error"
        assert data["type"] == "Exception"
        assert data["message"] == "An unexpected error occurred"  # Generic message, no leak
        assert "request_id" in data
        # Ensure we don't leak internal error details
        assert "RuntimeError" not in data["message"]
        assert "Internal error" not in data["message"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.integration
def test_error_response_format(client):
    """Test that error responses have consistent format."""
    # Trigger a 404 error (FastAPI default)
    response = client.get("/api/nonexistent")
    
    # FastAPI's default 404 handler returns {"detail": "Not Found"}
    # Our custom handlers return the structured format
    assert response.status_code == 404
    
    # Test with a route that exists but raises exception
    from src.api.dependencies import get_stock_data_service
    from src.application.services.stock_data_service import StockDataService
    
    mock_service = Mock(spec=StockDataService)
    mock_service.get_stock_quote = Mock(side_effect=ValidationError("Test error"))
    
    app.dependency_overrides[get_stock_data_service] = lambda: mock_service
    
    try:
        response = client.get("/api/stock/quote/TEST")
        
        assert response.status_code == 400
        data = response.json()
        # Verify structured error format
        assert "error" in data
        assert "message" in data
        assert "type" in data
        assert "request_id" in data
    finally:
        app.dependency_overrides.clear()

"""Integration tests for QA API endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, AsyncMock


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.integration
def test_qa_endpoint(client, mock_llm_provider, mock_vector_store, mock_embedding_provider):
    """Test POST /api/qa endpoint."""
    with patch('src.api.dependencies.get_llm_provider', return_value=mock_llm_provider), \
         patch('src.api.dependencies.get_vector_store', return_value=mock_vector_store), \
         patch('src.api.dependencies.get_embedding_service', return_value=mock_embedding_provider):
        
        mock_llm_provider.generate = AsyncMock(return_value="The answer is 1000 VND")
        
        response = client.post(
            "/api/qa",
            json={
                "question": "What is the EPS?",
                "context": "EPS: 1000 VND"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)


@pytest.mark.integration
def test_qa_endpoint_with_retrieved_docs(
    client, 
    mock_llm_provider, 
    mock_vector_store, 
    mock_embedding_provider
):
    """Test QA endpoint with vector store retrieval."""
    with patch('src.api.dependencies.get_llm_provider', return_value=mock_llm_provider), \
         patch('src.api.dependencies.get_vector_store', return_value=mock_vector_store), \
         patch('src.api.dependencies.get_embedding_service', return_value=mock_embedding_provider):
        
        # Mock vector store to return documents
        mock_vector_store.search = AsyncMock(return_value=[
            {
                "documentId": "doc-1",
                "source": "analysis_report",
                "sourceUrl": None,
                "title": "Báo cáo Q1",
                "section": "Tổng quan",
                "symbol": "ABC",
                "chunkId": "doc-1:0:0",
                "score": 0.9,
                "text": "EPS is 1000 VND"
            }
        ])
        mock_llm_provider.generate = AsyncMock(return_value="EPS is 1000 VND")
        
        response = client.post(
            "/api/qa",
            json={
                "question": "What is the EPS?",
                "context": "Base context"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["sources"], list)
        if data["sources"]:
            source_item = data["sources"][0]
            assert "documentId" in source_item
            assert "source" in source_item
            assert "sourceUrl" in source_item
            assert "title" in source_item
            assert "section" in source_item
            assert "symbol" in source_item
            assert "chunkId" in source_item
            assert "score" in source_item
            assert "textPreview" in source_item

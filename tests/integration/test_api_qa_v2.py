"""Integration tests for QA API v2 payload schema."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, AsyncMock


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.integration
def test_qa_v2_payload_schema(client, mock_llm_provider, mock_vector_store, mock_embedding_provider):
    """Test POST /api/qa with v2 payload fields."""
    with patch('src.api.dependencies.get_llm_provider', return_value=mock_llm_provider), \
         patch('src.api.dependencies.get_vector_store', return_value=mock_vector_store), \
         patch('src.api.dependencies.get_embedding_service', return_value=mock_embedding_provider):
        
        mock_vector_store.search = AsyncMock(return_value=[
            {
                "documentId": "doc-2",
                "source": "analysis_report",
                "sourceUrl": "https://example.com/report",
                "title": "Báo cáo Q2",
                "section": "Tài chính",
                "symbol": "XYZ",
                "chunkId": "doc-2:0:1",
                "score": 0.92,
                "text": "Doanh thu tăng 25% trong quý."
            }
        ])
        mock_llm_provider.generate = AsyncMock(return_value="Doanh thu tăng 25% trong quý.")
        
        response = client.post(
            "/api/qa",
            json={
                "question": "Kết quả kinh doanh quý thế nào?",
                "base_context": "Bối cảnh tổng quan Q2.",
                "top_k": 3,
                "document_id": "doc-2",
                "source": "analysis_report",
                "symbol": "XYZ"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data.get("answer"), str)
        assert isinstance(data.get("sources"), list)
        if data["sources"]:
            item = data["sources"][0]
            assert "documentId" in item
            assert "source" in item
            assert "sourceUrl" in item
            assert "title" in item
            assert "section" in item
            assert "symbol" in item
            assert "chunkId" in item
            assert "score" in item
            assert "textPreview" in item


@pytest.mark.integration
def test_qa_v2_backward_compat_context(client, mock_llm_provider, mock_vector_store, mock_embedding_provider):
    """Test backward compatibility with legacy context field."""
    with patch('src.api.dependencies.get_llm_provider', return_value=mock_llm_provider), \
         patch('src.api.dependencies.get_vector_store', return_value=mock_vector_store), \
         patch('src.api.dependencies.get_embedding_service', return_value=mock_embedding_provider):
        
        mock_vector_store.search = AsyncMock(return_value=[])
        mock_llm_provider.generate = AsyncMock(return_value="Không đủ dữ liệu.")
        
        response = client.post(
            "/api/qa",
            json={
                "question": "Thông tin gì?",
                "context": "Nguồn dữ liệu cũ."
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data.get("answer"), str)
        assert isinstance(data.get("sources"), list)

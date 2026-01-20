"""Integration tests for RAG ingest endpoint."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch
from src.api.main import app
from src.application.services.rag_ingest_service import RagIngestService
from src.domain.interfaces.vector_store import VectorStore
from src.domain.interfaces.embedding_provider import EmbeddingProvider


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.integration
def test_rag_ingest_endpoint(client):
    """Test POST /api/rag/ingest endpoint."""
    mock_vector_store = Mock(spec=VectorStore)
    mock_embedding_provider = Mock(spec=EmbeddingProvider)
    mock_embedding_provider.generate_embedding = AsyncMock(return_value=[0.1] * 8)
    mock_vector_store.upsert_chunks = AsyncMock(return_value=None)
    mock_vector_store.collection_name = "stock_documents"

    service = RagIngestService(mock_vector_store, mock_embedding_provider)

    with patch("src.api.dependencies.get_rag_ingest_service", return_value=service):
        response = client.post(
            "/api/rag/ingest",
            json={
                "document_id": "report-abc-2024",
                "source": "analysis_report",
                "text": "Đoạn 1.\n\nĐoạn 2 có nội dung dài hơn một chút.",
                "metadata": {
                    "sourceUrl": "https://example.com/report",
                    "title": "Báo cáo Q2",
                    "symbol": "VNM",
                    "section": "Tài chính"
                },
                "chunk_size": 50,
                "chunk_overlap": 10
            }
        )

    assert response.status_code == 200
    data = response.json()
    assert data["documentId"] == "report-abc-2024"
    assert data["chunksUpserted"] > 0
    assert data["collection"] == "stock_documents"
    assert data["status"] == "ok"

    assert mock_vector_store.upsert_chunks.called
    call_kwargs = mock_vector_store.upsert_chunks.call_args.kwargs
    payloads = call_kwargs["payloads"]
    assert isinstance(payloads, list)
    assert payloads
    payload = payloads[0]
    assert "documentId" in payload
    assert "source" in payload
    assert "sourceUrl" in payload
    assert "title" in payload
    assert "section" in payload
    assert "symbol" in payload
    assert "chunkId" in payload
    assert "text" in payload

"""Integration tests for RAG delete endpoint."""
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
def test_rag_delete_endpoint(client):
    """Test DELETE /api/rag/doc/{document_id} endpoint."""
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.delete_document = AsyncMock(return_value=3)
    mock_embedding_provider = Mock(spec=EmbeddingProvider)

    service = RagIngestService(mock_vector_store, mock_embedding_provider)

    with patch("src.api.dependencies.get_rag_ingest_service", return_value=service):
        response = client.delete("/api/rag/doc/report-abc-2024")

    assert response.status_code == 200
    data = response.json()
    assert data["documentId"] == "report-abc-2024"
    assert data["deleted"] == 3
    assert data["status"] == "ok"
    mock_vector_store.delete_document.assert_called_once_with("report-abc-2024")

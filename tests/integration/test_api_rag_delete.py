"""Integration tests for RAG delete endpoint."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock
from src.api.main import app
from src.application.services.rag_ingest_service import RagIngestService
from src.domain.interfaces.vector_store import VectorStore
from src.domain.interfaces.embedding_provider import EmbeddingProvider
from src.api import dependencies


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.integration
def test_rag_delete_endpoint(client, monkeypatch):
    """Test DELETE /api/rag/doc/{document_id} endpoint."""
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    from src.shared import config
    config._settings = None
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.delete_document = AsyncMock(return_value=17)
    mock_embedding_provider = Mock(spec=EmbeddingProvider)

    service = RagIngestService(mock_vector_store, mock_embedding_provider)

    app.dependency_overrides[dependencies.get_rag_ingest_service] = lambda: service
    try:
        response = client.delete("/api/rag/doc/report-abc-2024")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["documentId"] == "report-abc-2024"
    assert data["deleted"] == 17
    assert data["status"] == "ok"
    mock_vector_store.delete_document.assert_called_once_with("report-abc-2024")


@pytest.mark.integration
def test_rag_delete_nonexistent_document(client, monkeypatch):
    """Test delete for document that doesn't exist (should return 0 deleted)."""
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    from src.shared import config
    config._settings = None
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.delete_document = AsyncMock(return_value=0)
    mock_embedding_provider = Mock(spec=EmbeddingProvider)

    service = RagIngestService(mock_vector_store, mock_embedding_provider)

    app.dependency_overrides[dependencies.get_rag_ingest_service] = lambda: service
    try:
        response = client.delete("/api/rag/doc/nonexistent-doc")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["documentId"] == "nonexistent-doc"
    assert data["deleted"] == 0
    assert data["status"] == "ok"


@pytest.mark.integration
def test_rag_delete_without_api_key_when_not_configured(client, monkeypatch):
    """Test delete without API key when INTERNAL_API_KEY is not set (should succeed)."""
    # Ensure INTERNAL_API_KEY is not set
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    
    # Force reload settings
    from src.shared import config
    config._settings = None
    
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.delete_document = AsyncMock(return_value=5)
    mock_embedding_provider = Mock(spec=EmbeddingProvider)

    service = RagIngestService(mock_vector_store, mock_embedding_provider)

    app.dependency_overrides[dependencies.get_rag_ingest_service] = lambda: service
    try:
        response = client.delete("/api/rag/doc/test-doc")
    finally:
        app.dependency_overrides.clear()

    # Should succeed (200) when key is not configured
    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] == 5


@pytest.mark.integration
def test_rag_delete_missing_header_when_key_configured(client, monkeypatch):
    """Test delete without header when INTERNAL_API_KEY is set (should fail with 401)."""
    # Set INTERNAL_API_KEY
    monkeypatch.setenv("INTERNAL_API_KEY", "test-secret-key")
    
    # Force reload settings
    from src.shared import config
    config._settings = None
    
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.delete_document = AsyncMock(return_value=5)
    mock_embedding_provider = Mock(spec=EmbeddingProvider)

    service = RagIngestService(mock_vector_store, mock_embedding_provider)

    app.dependency_overrides[dependencies.get_rag_ingest_service] = lambda: service
    try:
        # Request WITHOUT header
        response = client.delete("/api/rag/doc/test-doc")
    finally:
        app.dependency_overrides.clear()

    # Should fail with 401
    assert response.status_code == 401


@pytest.mark.integration
def test_rag_delete_with_valid_header_when_key_configured(client, monkeypatch):
    """Test delete with correct header when INTERNAL_API_KEY is set (should succeed)."""
    # Set INTERNAL_API_KEY
    monkeypatch.setenv("INTERNAL_API_KEY", "test-secret-key")
    
    # Force reload settings
    from src.shared import config
    config._settings = None
    
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.delete_document = AsyncMock(return_value=5)
    mock_embedding_provider = Mock(spec=EmbeddingProvider)

    service = RagIngestService(mock_vector_store, mock_embedding_provider)

    app.dependency_overrides[dependencies.get_rag_ingest_service] = lambda: service
    try:
        # Request WITH correct header
        response = client.delete(
            "/api/rag/doc/test-doc",
            headers={"X-Internal-Api-Key": "test-secret-key"}
        )
    finally:
        app.dependency_overrides.clear()

    # Should succeed
    assert response.status_code == 200
    data = response.json()
    assert data["documentId"] == "test-doc"
    assert data["deleted"] == 5

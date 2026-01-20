"""Integration tests for RAG ingest endpoint."""
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
def test_rag_ingest_endpoint_with_chunk_params(client, monkeypatch):
    """Test POST /api/rag/ingest endpoint with custom chunk parameters."""
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    from src.shared import config
    config._settings = None
    mock_vector_store = Mock(spec=VectorStore)
    mock_embedding_provider = Mock(spec=EmbeddingProvider)
    mock_embedding_provider.generate_embedding = AsyncMock(return_value=[0.1] * 8)
    mock_vector_store.upsert_chunks = AsyncMock(return_value=None)
    mock_vector_store.collection_name = "stock_documents"

    service = RagIngestService(mock_vector_store, mock_embedding_provider)

    app.dependency_overrides[dependencies.get_rag_ingest_service] = lambda: service
    try:
        response = client.post(
            "/api/rag/ingest",
            json={
                "document_id": "report-abc-2024",
                "source": "analysis_report",
                "text": "Đoạn 1.\n\nĐoạn 2 có nội dung dài hơn một chút.\n\nĐoạn 3 thêm text.",
                "metadata": {
                    "sourceUrl": "https://example.com/report",
                    "title": "Báo cáo Q2",
                    "symbol": "VNM",
                    "section": "Tài chính"
                },
                "chunk_size": 400,
                "chunk_overlap": 80
            }
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["documentId"] == "report-abc-2024"
    assert data["chunksUpserted"] > 0
    assert data["collection"] == "stock_documents"
    assert data["status"] == "ok"

    # Verify payload structure
    assert mock_vector_store.upsert_chunks.called
    call_kwargs = mock_vector_store.upsert_chunks.call_args.kwargs
    payloads = call_kwargs["payloads"]
    assert isinstance(payloads, list)
    assert payloads
    payload = payloads[0]
    
    # Check camelCase keys
    assert "documentId" in payload
    assert "source" in payload
    assert "sourceUrl" in payload
    assert "title" in payload
    assert "section" in payload
    assert "symbol" in payload
    assert "chunkId" in payload
    assert "text" in payload


@pytest.mark.integration
def test_rag_ingest_endpoint_long_text(client, monkeypatch):
    """Test POST /api/rag/ingest with long text to ensure chunking doesn't crash."""
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    from src.shared import config
    config._settings = None
    mock_vector_store = Mock(spec=VectorStore)
    mock_embedding_provider = Mock(spec=EmbeddingProvider)
    mock_embedding_provider.generate_embedding = AsyncMock(return_value=[0.1] * 8)
    mock_vector_store.upsert_chunks = AsyncMock(return_value=None)
    mock_vector_store.collection_name = "stock_documents"

    service = RagIngestService(mock_vector_store, mock_embedding_provider)

    # Generate long text with multiple paragraphs
    long_text = "\n\n".join([f"Đoạn văn số {i}. " + "Nội dung " * 50 for i in range(20)])

    app.dependency_overrides[dependencies.get_rag_ingest_service] = lambda: service
    try:
        response = client.post(
            "/api/rag/ingest",
            json={
                "document_id": "long-doc-2024",
                "source": "analysis_report",
                "text": long_text,
                "metadata": {
                    "title": "Long Document",
                    "symbol": "ABC"
                }
            }
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["documentId"] == "long-doc-2024"
    assert data["chunksUpserted"] > 1  # Should create multiple chunks
    assert data["status"] == "ok"


@pytest.mark.integration
def test_rag_ingest_without_api_key_when_not_configured(client, monkeypatch):
    """Test ingest without API key when INTERNAL_API_KEY is not set (should succeed)."""
    # Ensure INTERNAL_API_KEY is not set
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    
    # Force reload settings to pick up env change
    from src.shared import config
    config._settings = None
    
    mock_vector_store = Mock(spec=VectorStore)
    mock_embedding_provider = Mock(spec=EmbeddingProvider)
    mock_embedding_provider.generate_embedding = AsyncMock(return_value=[0.1] * 8)
    mock_vector_store.upsert_chunks = AsyncMock(return_value=None)
    mock_vector_store.collection_name = "stock_documents"

    service = RagIngestService(mock_vector_store, mock_embedding_provider)

    app.dependency_overrides[dependencies.get_rag_ingest_service] = lambda: service
    try:
        # Request WITHOUT X-Internal-Api-Key header
        response = client.post(
            "/api/rag/ingest",
            json={
                "document_id": "test-doc",
                "source": "test",
                "text": "Test content",
                "metadata": {}
            }
        )
    finally:
        app.dependency_overrides.clear()

    # Should succeed (200) when key is not configured
    assert response.status_code == 200


@pytest.mark.integration
def test_rag_ingest_missing_header_when_key_configured(client, monkeypatch):
    """Test ingest without header when INTERNAL_API_KEY is set (should fail with 401)."""
    # Set INTERNAL_API_KEY
    monkeypatch.setenv("INTERNAL_API_KEY", "test-secret-key")
    
    # Force reload settings
    from src.shared import config
    config._settings = None
    
    mock_vector_store = Mock(spec=VectorStore)
    mock_embedding_provider = Mock(spec=EmbeddingProvider)
    mock_embedding_provider.generate_embedding = AsyncMock(return_value=[0.1] * 8)
    mock_vector_store.upsert_chunks = AsyncMock(return_value=None)
    mock_vector_store.collection_name = "stock_documents"

    service = RagIngestService(mock_vector_store, mock_embedding_provider)

    app.dependency_overrides[dependencies.get_rag_ingest_service] = lambda: service
    try:
        # Request WITHOUT header
        response = client.post(
            "/api/rag/ingest",
            json={
                "document_id": "test-doc",
                "source": "test",
                "text": "Test content",
                "metadata": {}
            }
        )
    finally:
        app.dependency_overrides.clear()

    # Should fail with 401
    assert response.status_code == 401
    assert "Missing X-Internal-Api-Key header" in response.json()["detail"]


@pytest.mark.integration
def test_rag_ingest_with_valid_header_when_key_configured(client, monkeypatch):
    """Test ingest with correct header when INTERNAL_API_KEY is set (should succeed)."""
    # Set INTERNAL_API_KEY
    monkeypatch.setenv("INTERNAL_API_KEY", "test-secret-key")
    
    # Force reload settings
    from src.shared import config
    config._settings = None
    
    mock_vector_store = Mock(spec=VectorStore)
    mock_embedding_provider = Mock(spec=EmbeddingProvider)
    mock_embedding_provider.generate_embedding = AsyncMock(return_value=[0.1] * 8)
    mock_vector_store.upsert_chunks = AsyncMock(return_value=None)
    mock_vector_store.collection_name = "stock_documents"

    service = RagIngestService(mock_vector_store, mock_embedding_provider)

    app.dependency_overrides[dependencies.get_rag_ingest_service] = lambda: service
    try:
        # Request WITH correct header
        response = client.post(
            "/api/rag/ingest",
            json={
                "document_id": "test-doc",
                "source": "test",
                "text": "Test content",
                "metadata": {}
            },
            headers={"X-Internal-Api-Key": "test-secret-key"}
        )
    finally:
        app.dependency_overrides.clear()

    # Should succeed
    assert response.status_code == 200
    data = response.json()
    assert data["documentId"] == "test-doc"


@pytest.mark.integration
def test_rag_ingest_invalid_chunk_params(client):
    """Test ingest with invalid chunk parameters (overlap >= size)."""
    response = client.post(
        "/api/rag/ingest",
        json={
            "document_id": "test-doc",
            "source": "test",
            "text": "Test content",
            "metadata": {},
            "chunk_size": 500,
            "chunk_overlap": 600  # Invalid: overlap >= size
        }
    )

    # Should fail with 422 validation error
    assert response.status_code == 422


@pytest.mark.integration
def test_rag_ingest_chunk_size_out_of_range(client):
    """Test ingest with chunk_size out of valid range."""
    response = client.post(
        "/api/rag/ingest",
        json={
            "document_id": "test-doc",
            "source": "test",
            "text": "Test content",
            "metadata": {},
            "chunk_size": 100  # Too small (< 300)
        }
    )

    # Should fail with 422 validation error
    assert response.status_code == 422

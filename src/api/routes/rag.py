"""RAG (Retrieval-Augmented Generation) API routes."""
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from src.application.services.rag_ingest_service import RagIngestService
from src.api.dependencies import get_rag_ingest_service
from src.shared.config import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

settings = get_settings()


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    document_id: str = Field(..., description="Unique document identifier (string)")
    source: str = Field(..., description="Source type (e.g., 'analysis_report')")
    text: str = Field(..., min_length=1, description="Full document text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class IngestResponse(BaseModel):
    """Response model for ingestion result."""
    chunksUpserted: int = Field(..., description="Number of chunks upserted")
    documentId: str = Field(..., description="Document identifier")
    collection: str = Field(..., description="Vector store collection name")
    embeddingModel: str = Field(..., description="Embedding model used")


def validate_api_key(x_internal_api_key: Optional[str] = Header(None)):
    """
    Validate internal API key header.
    
    Args:
        x_internal_api_key: API key from X-Internal-Api-Key header
        
    Raises:
        HTTPException: 401 if key is missing or invalid
    """
    # Get expected key from settings
    expected_key = getattr(settings, 'internal_api_key', None)
    
    # If no key configured in settings, check env directly
    if not expected_key:
        import os
        expected_key = os.getenv('INTERNAL_API_KEY')
    
    # Validate
    if not expected_key:
        logger.error("INTERNAL_API_KEY not configured in environment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal API key not configured"
        )
    
    if not x_internal_api_key:
        logger.warning("RAG ingest request missing X-Internal-Api-Key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Internal-Api-Key header"
        )
    
    if x_internal_api_key != expected_key:
        logger.warning("RAG ingest request with invalid X-Internal-Api-Key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-Internal-Api-Key"
        )
    
    # Valid key
    return True


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    service: RagIngestService = Depends(get_rag_ingest_service),
    _api_key_valid: bool = Depends(validate_api_key)
):
    """
    Ingest a document into RAG vector store.
    
    This endpoint:
    1. Chunks the document text (heading-first, then fixed-size with overlap)
    2. Generates embeddings for each chunk
    3. Upserts to Qdrant with deterministic point IDs and metadata
    
    Requires X-Internal-Api-Key header for authentication.
    
    Args:
        request: Ingest request with document data
        service: RAG ingest service instance
        _api_key_valid: API key validation result (injected)
        
    Returns:
        Ingest result with statistics
    """
    try:
        logger.info(
            f"Ingesting document {request.document_id}, source={request.source}, "
            f"text_length={len(request.text)}"
        )
        
        result = await service.ingest(
            document_id=request.document_id,
            source=request.source,
            text=request.text,
            metadata=request.metadata
        )
        
        logger.info(
            f"Successfully ingested document {request.document_id}: "
            f"{result['chunksUpserted']} chunks"
        )
        
        return IngestResponse(
            chunksUpserted=result["chunksUpserted"],
            documentId=result["documentId"],
            collection=result["collection"],
            embeddingModel=result["embeddingModel"]
        )
    
    except Exception as e:
        logger.error(f"Error ingesting document {request.document_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )

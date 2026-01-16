"""Qdrant vector store client implementation."""
from typing import List, Dict, Any
from qdrant_client import QdrantClient as Qdrant
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from src.domain.interfaces.vector_store import VectorStore
from src.shared.config import get_settings
from src.shared.exceptions import VectorStoreError
from src.shared.constants import DEFAULT_QDRANT_COLLECTION_NAME, DEFAULT_EMBEDDING_DIMENSION
from src.shared.logging import get_logger

logger = get_logger(__name__)


class QdrantClient(VectorStore):
    """Qdrant vector store client implementing VectorStore interface."""
    
    def __init__(self):
        """Initialize Qdrant client."""
        settings = get_settings()
        qdrant_url = settings.qdrant_url
        self.collection_name = settings.qdrant_collection_name or DEFAULT_QDRANT_COLLECTION_NAME
        
        try:
            self.client = Qdrant(url=qdrant_url)
            logger.info(f"Connected to Qdrant at {qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise VectorStoreError(f"Failed to connect to Qdrant: {str(e)}") from e

    async def search(
        self,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List of search results with text, source, and score
            
        Raises:
            VectorStoreError: If search operation fails
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            return [
                {
                    "text": hit.payload.get("text", ""),
                    "source": hit.payload.get("source", ""),
                    "score": hit.score
                }
                for hit in results
            ]
        except UnexpectedResponse as e:
            # Collection might not exist yet
            logger.warning(f"Collection {self.collection_name} does not exist: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error searching Qdrant: {str(e)}")
            raise VectorStoreError(f"Search operation failed: {str(e)}") from e

    async def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """
        Upsert vectors into the collection.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'vector', and 'payload'
            
        Raises:
            VectorStoreError: If upsert operation fails
        """
        try:
            # Check if collection exists, create if not
            try:
                self.client.get_collection(self.collection_name)
            except UnexpectedResponse:
                # Collection doesn't exist, create it
                settings = get_settings()
                dimension = settings.embedding_dimension or DEFAULT_EMBEDDING_DIMENSION
                
                logger.info(f"Creating collection {self.collection_name} with dimension {dimension}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            
            # Upsert vectors
            self.client.upsert(
                collection_name=self.collection_name,
                points=vectors
            )
            logger.debug(f"Upserted {len(vectors)} vectors to {self.collection_name}")
        except Exception as e:
            logger.error(f"Error upserting vectors to Qdrant: {str(e)}")
            raise VectorStoreError(f"Upsert operation failed: {str(e)}") from e

import os
from qdrant_client import QdrantClient as Qdrant
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()


class QdrantClient:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = Qdrant(url=qdrant_url)
        self.collection_name = "stock_documents"

    async def search(self, query_vector: list[float], limit: int = 5) -> list[dict]:
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
        except Exception as e:
            # Collection might not exist yet
            return []

    async def upsert(self, vectors: list[dict]):
        # Initialize collection if it doesn't exist
        try:
            self.client.get_collection(self.collection_name)
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        
        # Upsert vectors
        self.client.upsert(
            collection_name=self.collection_name,
            points=vectors
        )


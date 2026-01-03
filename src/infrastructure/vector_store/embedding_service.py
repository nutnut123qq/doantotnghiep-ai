from sentence_transformers import SentenceTransformer
import asyncio


class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    async def generate_embedding(self, text: str) -> list[float]:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self.model.encode, text)
        return embedding.tolist()


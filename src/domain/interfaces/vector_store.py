from abc import ABC, abstractmethod
from typing import List, Dict


class VectorStore(ABC):
    @abstractmethod
    async def search(self, query_vector: List[float], limit: int = 5) -> List[Dict]:
        pass

    @abstractmethod
    async def upsert(self, vectors: List[Dict]):
        pass


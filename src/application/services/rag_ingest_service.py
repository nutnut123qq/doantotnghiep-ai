"""RAG ingest service for chunking and embedding documents into Qdrant."""
import re
from typing import Dict, Any, List, Optional, Tuple
from src.domain.interfaces.vector_store import VectorStore
from src.domain.interfaces.embedding_provider import EmbeddingProvider
from src.shared.logging import get_logger

logger = get_logger(__name__)

# Chunking constants
CHUNK_SIZE_CHARS = 1500
OVERLAP_CHARS = 200
MIN_CHUNK_SIZE = 1200
MAX_CHUNK_SIZE = 1800

# Heading detection patterns (priority order)
HEADING_PATTERNS = [
    (r'^(SECTION:|Section:)\s+(.+)$', 'SECTION'),  # Highest priority
    (r'^(#{1,6})\s+(.+)$', 'MARKDOWN'),
    (r'^(\d+(?:\.\d+)*)\s+(.+)$', 'NUMBERED'),
    (r'^([IVXLCDM]+)\.\s+(.+)$', 'ROMAN'),
]


class RagIngestService:
    """Service for ingesting documents into RAG vector store."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider
    ):
        """
        Initialize RAG ingest service.
        
        Args:
            vector_store: Vector store for storing embeddings
            embedding_provider: Provider for generating embeddings
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        logger.info("Initialized RagIngestService")
    
    async def ingest(
        self,
        document_id: str,
        source: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ingest a document by chunking, embedding, and upserting to vector store.
        
        Args:
            document_id: Unique document identifier (string)
            source: Source type (e.g., "analysis_report")
            text: Full document text
            metadata: Document metadata (symbol, title, sourceUrl, etc.)
            
        Returns:
            Ingest result with stats
        """
        logger.info(f"Starting ingest for document {document_id}, source={source}")
        
        # Extract metadata fields (handle both camelCase and snake_case)
        symbol = metadata.get("symbol")
        title = metadata.get("title")
        source_url = metadata.get("sourceUrl") or metadata.get("source_url")
        
        # Split into sections based on headings
        sections = self._split_into_sections(text)
        logger.debug(f"Split document into {len(sections)} sections")
        
        # Chunk each section
        all_chunks = []
        for section_idx, (section_title, section_text) in enumerate(sections):
            chunks = self._chunk_section(section_text)
            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "section_idx": section_idx,
                    "chunk_idx": chunk_idx,
                    "section_title": section_title,
                    "text": chunk_text
                })
        
        logger.info(f"Created {len(all_chunks)} chunks for document {document_id}")
        
        # Embed and upsert each chunk
        points = []
        for chunk_data in all_chunks:
            # Generate embedding
            embedding = await self.embedding_provider.generate_embedding(chunk_data["text"])
            
            # Create deterministic point ID
            point_id = f"{document_id}:{chunk_data['section_idx']}:{chunk_data['chunk_idx']}"
            
            # Build payload (camelCase keys)
            payload = {
                "text": chunk_data["text"],
                "source": source,
                "documentId": document_id,
                "symbol": symbol,
                "section": chunk_data["section_title"],
                "title": title,
                "sourceUrl": source_url
            }
            
            points.append({
                "id": point_id,
                "vector": embedding,
                "payload": payload
            })
        
        # Upsert to vector store
        await self.vector_store.upsert(points)
        
        logger.info(f"Successfully upserted {len(points)} chunks for document {document_id}")
        
        # Get collection name and embedding model from settings
        collection_name = getattr(self.vector_store, 'collection_name', 'stock_documents')
        embedding_model = getattr(
            self.embedding_provider,
            'model_name',
            'all-MiniLM-L6-v2'
        )
        
        return {
            "chunksUpserted": len(points),
            "documentId": document_id,
            "collection": collection_name,
            "embeddingModel": embedding_model
        }
    
    def _split_into_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into sections based on heading patterns.
        
        Args:
            text: Full document text
            
        Returns:
            List of (section_title, section_text) tuples
        """
        lines = text.split('\n')
        sections = []
        current_section_title = "Introduction"
        current_section_lines = []
        
        for line in lines:
            # Check if line matches any heading pattern (in priority order)
            is_heading = False
            for pattern, pattern_type in HEADING_PATTERNS:
                match = re.match(pattern, line.strip())
                if match:
                    # Save previous section if has content
                    if current_section_lines:
                        section_text = '\n'.join(current_section_lines).strip()
                        if section_text:
                            sections.append((current_section_title, section_text))
                    
                    # Start new section
                    current_section_title = line.strip()
                    current_section_lines = []
                    is_heading = True
                    break
            
            if not is_heading:
                current_section_lines.append(line)
        
        # Add final section
        if current_section_lines:
            section_text = '\n'.join(current_section_lines).strip()
            if section_text:
                sections.append((current_section_title, section_text))
        
        # If no sections found (no headings), return entire text as one section
        if not sections:
            sections.append(("Content", text))
        
        return sections
    
    def _chunk_section(self, text: str) -> List[str]:
        """
        Chunk section text into overlapping chunks.
        
        Args:
            text: Section text
            
        Returns:
            List of chunk texts
        """
        # If section is smaller than max chunk size, return as single chunk
        if len(text) <= MAX_CHUNK_SIZE:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position (clamp to min/max)
            end = start + CHUNK_SIZE_CHARS
            
            # If this is not the last chunk, try to break at sentence/word boundary
            if end < len(text):
                # Look for sentence end (., !, ?) within last 20% of chunk
                search_start = end - int(CHUNK_SIZE_CHARS * 0.2)
                sentence_end = max(
                    text.rfind('. ', search_start, end),
                    text.rfind('! ', search_start, end),
                    text.rfind('? ', search_start, end)
                )
                if sentence_end > search_start:
                    end = sentence_end + 1
                else:
                    # Try word boundary
                    space_pos = text.rfind(' ', search_start, end)
                    if space_pos > search_start:
                        end = space_pos
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - OVERLAP_CHARS
            
            # Ensure we make progress
            if start <= chunks[-1] if chunks else 0:
                start = end
        
        return chunks

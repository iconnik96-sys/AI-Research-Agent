import re
from typing import Dict, List, Optional

from app.core.config import settings
from app.schemas.chunk import DocumentChunk
from app.schemas.document import Document


class TextChunker:
    """Deterministic document chunker with configurable size and overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def normalize_text(text: str) -> str:
        """Safely clean and normalize text before chunking."""
        if not text:
            return ""
        # Remove null bytes which cause PostgreSQL text issues
        text = text.replace("\x00", "")
        # Normalize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple horizontal whitespaces
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse excessive newlines (max 2 consecutive newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_text(self, text: str) -> List[str]:
        """Split normalized text into deterministic chunks."""
        normalized = self.normalize_text(text)
        if not normalized:
            return []

        # Handle short text: single chunk
        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: List[str] = []
        start = 0
        text_len = len(normalized)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                # Seek a natural break boundary (paragraph break, line break, or space)
                lookback = max(start, end - min(self.chunk_overlap, 100))
                break_found = -1
                for sep in ("\n\n", "\n", " "):
                    idx = normalized.rfind(sep, lookback, end)
                    if idx != -1:
                        break_found = idx + len(sep)
                        break
                if break_found > start:
                    end = break_found

            chunk_content = normalized[start:end].strip()
            if chunk_content:
                chunks.append(chunk_content)

            if end >= text_len:
                break

            # Advance start index respecting overlap
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks

    def chunk_document(
        self,
        document: Document,
        document_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Chunk an extracted Document into ordered DocumentChunk models."""
        raw_chunks = self.split_text(document.text)
        return [
            DocumentChunk(
                document_id=document_id,
                session_id=session_id,
                chunk_index=i,
                text=chunk_text,
                char_count=len(chunk_text),
            )
            for i, chunk_text in enumerate(raw_chunks)
        ]

    def chunk_documents(
        self,
        documents: List[Document],
        document_id_map: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Chunk a list of extracted Documents, linking document_id and session_id."""
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            doc_id = document_id_map.get(doc.url) if document_id_map else None
            chunks = self.chunk_document(doc, document_id=doc_id, session_id=session_id)
            all_chunks.extend(chunks)
        return all_chunks


def get_text_chunker() -> TextChunker:
    """Dependency provider for TextChunker."""
    return TextChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

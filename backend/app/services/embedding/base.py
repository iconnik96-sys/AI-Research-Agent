from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract interface for embedding generation."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the vector dimensionality of this embedding provider."""
        pass

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, each matching `dimensions` in length.
        """
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate a vector embedding for a single text string.

        Args:
            text: String to embed.

        Returns:
            A float vector matching `dimensions` in length.
        """
        pass

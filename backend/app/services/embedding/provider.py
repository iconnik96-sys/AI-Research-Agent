import logging
from typing import List, Optional
import httpx

from app.core.config import settings
from app.services.embedding.base import BaseEmbeddingProvider
from app.services.embedding.exceptions import (
    EmbeddingConfigError,
    EmbeddingNetworkError,
    EmbeddingProviderError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider communicating with OpenAI-compatible embedding APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.EMBEDDING_API_KEY
        self.base_url = base_url if base_url is not None else settings.EMBEDDING_BASE_URL
        self.model = model if model is not None else settings.EMBEDDING_MODEL
        self._dimensions = (
            dimensions
            if dimensions is not None
            else settings.EMBEDDING_DIMENSIONS
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.EMBEDDING_TIMEOUT_SECONDS
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text."""
        results = await self.embed_texts([text])
        if not results:
            raise EmbeddingResponseError("No embedding vector returned for text.")
        return results[0]

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of texts."""
        if not texts:
            return []

        # Validate credentials: if not a local address, require API key
        is_local = "localhost" in self.base_url.lower() or "127.0.0.1" in self.base_url.lower()
        if not is_local and (not self.api_key or not self.api_key.strip()):
            raise EmbeddingConfigError(
                "Embedding API key is not configured. Please set EMBEDDING_API_KEY in your environment."
            )

        endpoint = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise EmbeddingTimeoutError(
                f"Embedding request timed out after {self.timeout_seconds}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingProviderError(
                f"Embedding provider error HTTP {exc.response.status_code}: {exc.response.text}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise EmbeddingNetworkError(
                f"Network error while connecting to embedding provider: {str(exc)}"
            ) from exc

        return self._parse_and_validate_response(data=data, expected_count=len(texts))

    def _parse_and_validate_response(
        self,
        data: dict,
        expected_count: int,
    ) -> List[List[float]]:
        if not isinstance(data, dict):
            raise EmbeddingResponseError("Embedding response must be a JSON object.")

        items = data.get("data")
        if not isinstance(items, list):
            raise EmbeddingResponseError("Embedding response missing valid 'data' array.")

        if len(items) != expected_count:
            raise EmbeddingResponseError(
                f"Embedding count mismatch: expected {expected_count}, received {len(items)}"
            )

        # Sort items by index to preserve input ordering
        sorted_items = sorted(items, key=lambda x: x.get("index", 0))

        embeddings: List[List[float]] = []
        for idx, item in enumerate(sorted_items):
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise EmbeddingResponseError(f"Embedding item {idx} is not a valid list.")
            if len(emb) != self._dimensions:
                raise EmbeddingResponseError(
                    f"Embedding dimension mismatch: expected {self._dimensions}, got {len(emb)}"
                )
            embeddings.append(emb)

        return embeddings

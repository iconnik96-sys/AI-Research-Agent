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


class SupabaseEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using Supabase Edge Function with built-in Supabase.ai (gte-small)."""

    def __init__(
        self,
        function_url: Optional[str] = None,
        anon_key: Optional[str] = None,
        supabase_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        batch_size: int = 4,
    ):
        self.supabase_url = (
            supabase_url if supabase_url is not None else settings.SUPABASE_URL
        )
        self.function_url = (
            function_url
            if function_url is not None
            else settings.SUPABASE_EMBEDDING_FUNCTION_URL
        )
        if not self.function_url and self.supabase_url:
            self.function_url = f"{self.supabase_url.rstrip('/')}/functions/v1/embed"

        self.anon_key = anon_key if anon_key is not None else settings.SUPABASE_ANON_KEY
        self.model = "gte-small"
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
        self.batch_size = batch_size if batch_size > 0 else 4

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_text(self, text: str) -> List[float]:
        """Generate 384-dimensional embedding vector for a single query text."""
        results = await self.embed_texts([text])
        if not results:
            raise EmbeddingResponseError("No embedding vector returned for text.")
        return results[0]

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate 384-dimensional embedding vectors for a list of texts via Supabase Edge Function."""
        if not texts:
            return []

        if not self.function_url or not self.function_url.strip():
            raise EmbeddingConfigError(
                "Supabase embedding function URL is not configured. "
                "Please set SUPABASE_EMBEDDING_FUNCTION_URL or SUPABASE_URL in your environment."
            )

        headers = {
            "Content-Type": "application/json",
        }
        if self.anon_key and self.anon_key.strip():
            headers["Authorization"] = f"Bearer {self.anon_key}"
            headers["apikey"] = self.anon_key

        all_embeddings: List[List[float]] = []

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i : i + self.batch_size]
                    payload = {
                        "input": batch,
                    }
                    response = await client.post(self.function_url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    batch_embeddings = self._parse_and_validate_response(
                        data=data, expected_count=len(batch)
                    )
                    all_embeddings.extend(batch_embeddings)
        except httpx.TimeoutException as exc:
            raise EmbeddingTimeoutError(
                f"Embedding request to Supabase timed out after {self.timeout_seconds}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingProviderError(
                f"Supabase embedding function error HTTP {exc.response.status_code}: {exc.response.text}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise EmbeddingNetworkError(
                f"Network error while connecting to Supabase embedding function: {str(exc)}"
            ) from exc

        return all_embeddings

    def _parse_and_validate_response(
        self,
        data: dict,
        expected_count: int,
    ) -> List[List[float]]:
        if not isinstance(data, dict):
            raise EmbeddingResponseError("Supabase embedding response must be a JSON object.")

        items = data.get("embeddings")
        if not isinstance(items, list):
            if "embedding" in data and isinstance(data["embedding"], list):
                items = [data["embedding"]]
            elif "data" in data and isinstance(data["data"], list):
                items = [item.get("embedding") if isinstance(item, dict) else item for item in data["data"]]
            else:
                raise EmbeddingResponseError("Supabase embedding response missing valid 'embeddings' array.")

        if len(items) != expected_count:
            raise EmbeddingResponseError(
                f"Embedding count mismatch: expected {expected_count}, received {len(items)}"
            )

        embeddings: List[List[float]] = []
        for idx, emb in enumerate(items):
            if not isinstance(emb, list):
                raise EmbeddingResponseError(f"Embedding item {idx} is not a valid list.")
            if len(emb) != self._dimensions:
                raise EmbeddingResponseError(
                    f"Embedding dimension mismatch: expected {self._dimensions}, got {len(emb)}"
                )
            embeddings.append(emb)

        return embeddings

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

from app.core.config import settings
from app.schemas.chunk import DocumentChunk
from app.schemas.document import Document

# Tokenization regex: matches alphanumeric words and compound terms with hyphens or slashes
# e.g., "solid-state", "lithium-ion", "wh/kg", "100/kwh", "2027-2030"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")

STOP_WORDS: Set[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
})


def tokenize(text: str) -> List[str]:
    """Tokenize text into normalized lowercase tokens, preserving compounds and their components."""
    if not text:
        return []
    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    tokens: List[str] = []
    for tok in raw_tokens:
        tokens.append(tok)
        # If compound term (e.g. solid-state or km/h), also index individual subparts
        if "-" in tok or "/" in tok:
            subparts = re.split(r"[-/]", tok)
            for sub in subparts:
                if sub and len(sub) > 1:
                    tokens.append(sub)
    return tokens


class CandidateChunkFilter:
    """Zero-dependency, pure-Python lexical BM25 candidate filter with document diversity capping."""

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.65,
        default_max_candidates: Optional[int] = None,
        default_max_per_doc: Optional[int] = None,
    ):
        self.k1 = k1
        self.b = b
        self.default_max_candidates = (
            default_max_candidates
            if default_max_candidates is not None
            else settings.CANDIDATE_CHUNKS_PRE_EMBED
        )
        self.default_max_per_doc = (
            default_max_per_doc
            if default_max_per_doc is not None
            else settings.CANDIDATE_CHUNKS_MAX_PER_DOC
        )

    def filter_candidates(
        self,
        chunks: List[DocumentChunk],
        documents: Optional[List[Document]] = None,
        queries: Optional[List[str]] = None,
        max_candidates: Optional[int] = None,
        max_per_doc: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """Filter and rank candidate chunks before remote embedding using multi-query BM25.

        Args:
            chunks: All raw DocumentChunk objects created from extracted documents.
            documents: Optional list of parent Document objects.
            queries: The original research question + all decomposed sub-questions.
            max_candidates: Maximum number of candidate chunks to select for embedding.
            max_per_doc: Maximum number of chunks allowed from any single document.

        Returns:
            A deterministic, shortlisted list of original DocumentChunk objects.
        """
        if not chunks:
            return []

        effective_max_candidates = (
            max_candidates if max_candidates is not None else self.default_max_candidates
        )
        effective_max_per_doc = (
            max_per_doc if max_per_doc is not None else self.default_max_per_doc
        )

        # If total chunks is already at or below target candidate count, return all chunks unmodified
        if len(chunks) <= effective_max_candidates:
            return list(chunks)

        # If no queries provided, return the earliest chunks up to max_candidates
        if not queries or not any(q.strip() for q in queries):
            return chunks[:effective_max_candidates]

        # Filter and normalize query token lists (strip stop words)
        clean_queries: List[List[str]] = []
        for q in queries:
            if not q or not q.strip():
                continue
            q_toks = [t for t in tokenize(q) if t not in STOP_WORDS and len(t) > 1]
            if q_toks:
                clean_queries.append(q_toks)

        if not clean_queries:
            return chunks[:effective_max_candidates]

        # Pre-tokenize all chunks and calculate corpus statistics
        n_docs = len(chunks)
        chunk_token_counts: List[Counter] = []
        chunk_lengths: List[int] = []
        doc_freq: Dict[str, int] = defaultdict(int)

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            chunk_token_counts.append(counts)
            chunk_lengths.append(len(tokens))
            for term in counts.keys():
                doc_freq[term] += 1

        avg_doc_len = sum(chunk_lengths) / max(n_docs, 1)

        # Precompute IDF for all terms in queries
        idf_cache: Dict[str, float] = {}
        for q_toks in clean_queries:
            for term in q_toks:
                if term not in idf_cache:
                    df = doc_freq.get(term, 0)
                    # Standard BM25 IDF formulation with smoothing
                    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                    idf_cache[term] = max(idf, 0.0)

        # Score each chunk against every query independently
        # Composite score = max(BM25 across queries) + 0.2 * sum(BM25 across queries)
        chunk_scores: List[float] = []
        for idx in range(n_docs):
            counts = chunk_token_counts[idx]
            doc_len = chunk_lengths[idx]
            len_norm = 1.0 - self.b + self.b * (doc_len / max(avg_doc_len, 1.0))

            per_query_scores: List[float] = []
            for q_toks in clean_queries:
                q_score = 0.0
                for term in q_toks:
                    tf = counts.get(term, 0)
                    if tf > 0:
                        idf = idf_cache.get(term, 0.0)
                        num = tf * (self.k1 + 1.0)
                        denom = tf + self.k1 * len_norm
                        q_score += idf * (num / denom)
                per_query_scores.append(q_score)

            if per_query_scores:
                max_s = max(per_query_scores)
                sum_s = sum(per_query_scores)
                composite_score = max_s + 0.2 * sum_s
            else:
                composite_score = 0.0

            chunk_scores.append(composite_score)

        # Pair chunks with their scores and original index for deterministic ordering
        # Sort key: (-score, str(document_id), chunk_index)
        scored_pairs = [
            (chunk_scores[i], chunks[i])
            for i in range(n_docs)
        ]
        scored_pairs.sort(
            key=lambda pair: (
                -pair[0],
                str(pair[1].document_id or ""),
                pair[1].chunk_index,
            )
        )

        selected: List[DocumentChunk] = []
        doc_chunk_counts: Dict[str, int] = defaultdict(int)
        overflow_with_score: List[DocumentChunk] = []
        zero_score_chunks: List[DocumentChunk] = []

        # Pass 1: Select top-scoring chunks respecting max_per_doc
        for score, chunk in scored_pairs:
            doc_key = str(chunk.document_id or "unknown")
            if score > 0.0:
                if doc_chunk_counts[doc_key] < effective_max_per_doc:
                    if len(selected) < effective_max_candidates:
                        selected.append(chunk)
                        doc_chunk_counts[doc_key] += 1
                    else:
                        break
                else:
                    overflow_with_score.append(chunk)
            else:
                zero_score_chunks.append(chunk)

        # Pass 2 (Fallback): If candidate slots remain, fill from overflow chunks that had positive BM25 scores
        if len(selected) < effective_max_candidates and overflow_with_score:
            for chunk in overflow_with_score:
                if len(selected) >= effective_max_candidates:
                    break
                selected.append(chunk)

        # Pass 3 (Fallback): If still below target, fill from remaining zero-score chunks
        if len(selected) < effective_max_candidates and zero_score_chunks:
            for chunk in zero_score_chunks:
                if len(selected) >= effective_max_candidates:
                    break
                selected.append(chunk)

        return selected


# Singleton filter instance
_default_candidate_filter = CandidateChunkFilter()


def filter_candidate_chunks(
    chunks: List[DocumentChunk],
    documents: Optional[List[Document]] = None,
    queries: Optional[List[str]] = None,
    max_candidates: Optional[int] = None,
    max_per_doc: Optional[int] = None,
) -> List[DocumentChunk]:
    """Convenience functional interface for candidate chunk filtering."""
    return _default_candidate_filter.filter_candidates(
        chunks=chunks,
        documents=documents,
        queries=queries,
        max_candidates=max_candidates,
        max_per_doc=max_per_doc,
    )

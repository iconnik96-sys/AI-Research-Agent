<div align="center">

# 🔬 AI Research Agent

### An autonomous, evidence-grounded research system

*Decomposes complex questions into structured search plans, gathers live web content, indexes vector embeddings with pgvector, independently verifies factual claims, and synthesizes cited research reports.*

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Supabase](https://img.shields.io/badge/Supabase-Database%20%26%20Edge%20AI-3ECF8E.svg?logo=supabase&logoColor=white)](https://supabase.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

</div>

<br>

## 📑 Table of Contents

<table>
<tr>
<td valign="top" width="33%">

**Overview**
- [1. Project Overview](#1-project-overview)
- [2. Key Features](#2-key-features)
- [3. Architecture](#3-architecture)
- [4. Research Pipeline](#4-research-pipeline)
- [5. Claim Verification Pipeline](#5-claim-verification-pipeline)
- [6. Retrieval, Chunking & Embeddings](#6-retrieval-chunking--embeddings)
- [7. Concurrency & Performance Engineering](#7-concurrency--performance-engineering)
- [8. Evaluation & Benchmark Suite](#8-evaluation--benchmark-suite)

</td>
<td valign="top" width="33%">

**Build & Run**
- [9. Tech Stack](#9-tech-stack)
- [10. Project Structure](#10-project-structure)
- [11. API Documentation](#11-api-documentation)
- [12. Local Setup & Installation](#12-local-setup--installation)
- [13. Environment Variables](#13-environment-variables)
- [14. Running the Application](#14-running-the-application)
- [15. Example Research Walkthrough](#15-example-research-walkthrough)

</td>
<td valign="top" width="33%">

**Engineering Notes**
- [16. Engineering Challenges & Solutions](#16-engineering-challenges--solutions)
- [17. Key Technical Decisions](#17-key-technical-decisions)
- [18. Security & Reliability Measures](#18-security--reliability-measures)
- [19. Cost & Free-Tier Architecture](#19-cost--free-tier-architecture)
- [20. Future Roadmap](#20-future-roadmap)
- [21. Why This Project Matters](#21-why-this-project-matters)
- [22. Interview Discussion Points](#22-interview-discussion-points)
- [23. Resume Description](#23-resume-description)

</td>
</tr>
</table>

---

## 1. Project Overview

### 🚩 The Problem
When queried with complex or open-ended analytical questions, standard Large Language Models (LLMs) suffer from fundamental failure modes:

1. **Hallucinations & Confabulation** — Generative models produce convincing yet factually false assertions, statistics, and citations.
2. **Outdated Knowledge Cutoffs** — Pre-training data is frozen at training time and cannot capture contemporary discoveries or technical literature.
3. **Lack of Source Traceability** — Traditional chat interfaces output unstructured paragraphs without verifiable links between individual assertions and primary source documents.
4. **Shallow Web Search** — Naive retrieval-augmented generation (RAG) issues a single search query, ingests noisy HTML text, and feeds unverified excerpts straight into the generation context.
5. **Conflicting Evidence** — Web sources frequently contradict one another. Unchecked LLM synthesis merges contradictory claims without evaluating evidence support.

### ✅ The Solution
The **AI Research Agent** is an evidence-grounded research pipeline designed to address these failure modes. Given a natural-language research question, the system:

- Decomposes the prompt into focused semantic sub-questions and targeted web queries.
- Concurrently searches the web, deduplicates URLs, and extracts clean text from HTML pages and binary PDF documents.
- Applies a local **BM25 lexical candidate filter** with document-diversity capping to prune noise before calling embedding models.
- Generates 384-dimensional vector embeddings on **Supabase Edge Functions** using native on-edge AI inference (`gte-small`).
- Stores vectors in **Supabase PostgreSQL** via `pgvector` and executes multi-query cosine similarity retrieval.
- Extracts candidate atomic factual claims and **executes a dedicated verification pass** against gathered evidence before report synthesis.
- Generates a structured markdown research report where **every section cites verified sources, citations are strictly validated, and unsupported claims are flagged**.

---

## 2. Key Features

| Feature | Description |
|---|---|
| 🔎 **Multi-Angle Research Planning** | Decomposes complex questions into up to 5 focused sub-questions and targeted search queries using structured LLM schema validation, ensuring broad exploration of technical nuances. |
| 🌐 **Concurrent Web Search & Multi-Format Extraction** | Searches the web via Tavily API with bounded concurrency. Asynchronously extracts, cleans, and normalizes text from HTML pages (stripping scripts, styles, navigation) and binary PDF documents (page-by-page parsing via `pypdf`). |
| ⚡ **BM25 Candidate Pre-Filtering & Diversity Capping** | Implements a pure-Python lexical BM25 ranking filter that scores all raw chunks against combined queries before remote embedding. Caps candidates at 60 total chunks and enforces a maximum of 6 chunks per document to maintain source diversity. |
| 🧠 **On-Edge Vector Embeddings (`gte-small`)** | Leverages native `Supabase.ai` edge inference running `gte-small` (384 dimensions) inside a Deno Edge Function, running within Supabase free-tier invocation quotas without third-party embedding API fees. |
| 🗄️ **PostgreSQL + pgvector Semantic Retrieval** | Stores document chunks with vector embeddings in PostgreSQL. Performs multi-query cosine similarity retrieval across both original and sub-queries, merging and deduplicating chunks by maximum similarity. |
| 🧾 **Atomic Factual Claim Extraction** | Extracts discrete, verifiable factual assertions from retrieved evidence items before report drafting. Each claim is mapped to candidate evidence IDs (`E1`, `E2`, ...). |
| ✅ **Dedicated Claim Verification Pass** | Cross-examines extracted claims against evidence excerpts via a **single-request batched verification prompt** with strict JSON schema validation, backed by an automated fallback to **semaphore-bounded concurrent individual verifications** (`concurrency=5`). |
| 📚 **Grounded Synthesis & Strict Citation Validation** | Synthesizes a structured report where sections cite exact Source IDs (`S1`, `S2`, ...). A validation gate verifies that every citation maps to an ingested document; invalid or hallucinated citations raise an explicit error. |
| 📊 **Standardized 15-Case Benchmark Evaluation Suite** | Includes an evaluation harness with a standardized 15-question benchmark dataset (`research_benchmark_v1.json`). Quantifies Planner Dimension Coverage, Retrieval Hit Rate, Average Similarity, Claim Support Rates, and Citation Precision. |
| 🛡️ **Security & Reliability Measures** | Built with `X-Request-ID` correlation across async workflows, security headers (`nosniff`, `DENY`), sliding-window in-memory rate limiting, sensitive data log filtering (credential and connection string sanitization), and PostgreSQL Row Level Security (RLS) default-deny policies. |
| 💻 **Responsive Dark-Mode Frontend** | Crafted with React 18, Vite, TypeScript, and Tailwind CSS. Features real-time elapsed query timers, interactive Research Dossier tabs (Report, Claims, Sources, Plan), and clickable citation badges that highlight referenced sources. |

#### Claim Verification Statuses
| Status | Meaning | Grounding Behavior in Synthesis |
|---|---|---|
| `SUPPORTED` | Direct, explicit confirmation in primary source text. | Synthesized as verified factual content. |
| `PARTIALLY_SUPPORTED` | Aspects confirmed, but details missing or qualified. | Synthesized with explicit qualification. |
| `UNSUPPORTED` | Contradicted by evidence. | Excluded or explicitly flagged as contradicted. |
| `INSUFFICIENT_EVIDENCE` | Provided text lacks conclusive data. | Excluded or communicated with explicit uncertainty. |

---

## 3. Architecture

The system is architected as an **asynchronous, deterministic research pipeline** coordinated by a single orchestrator service, rather than an unconstrained multi-agent loop:

```mermaid
flowchart TD
    subgraph Client ["Frontend (React 18 + Vite + TypeScript)"]
        UI["User Interface"]
        Input["Research Input"]
        Progress["Live Pipeline Progress"]
        Dossier["Research Dossier (Report / Claims / Sources / Plan)"]
    end

    subgraph API ["FastAPI Backend (Python 3.11 async)"]
        Router["API Router (/api/research)"]
        MW1["RequestID Middleware (X-Request-ID)"]
        MW2["Security Headers Middleware"]
        MW3["In-Memory Rate Limiter Middleware"]
        Orchestrator["Research Orchestrator"]
    end

    subgraph Planning ["Stage 1: Planning"]
        Planner["LLM Research Planner (Configurable Model)"]
        SubQ["Decomposed Sub-Questions & Search Queries"]
    end

    subgraph Ingestion ["Stage 2: Web Search & Extraction"]
        Tavily["Tavily Search API (Concurrent Semaphore)"]
        Scraper["Async Content Extractor"]
        HTML["BeautifulSoup HTML Sanitizer"]
        PDF["PyPDF Binary Extractor"]
    end

    subgraph Processing ["Stage 3: Chunking & Pre-Filtering"]
        Chunker["Sliding Window Chunker (1000 chars, 200 overlap)"]
        BM25["BM25 Candidate Filter (Max 60 chunks, Max 6/doc)"]
    end

    subgraph VectorAI ["Stage 4: Edge Embeddings & Vector Search"]
        EdgeEmbed["Supabase Edge Function (gte-small, 384-dim)"]
        PgVector["Supabase PostgreSQL (pgvector cosine similarity)"]
        MultiRetriever["Multi-Query Semantic Retriever"]
    end

    subgraph Verification ["Stage 5: Claim Verification"]
        ClaimExtract["Atomic Claim Extractor"]
        Verifier["Claim Verifier (Primary: Batched / Fallback: Semaphore)"]
        ClaimDB["Claims & Evidence Junction Tables"]
    end

    subgraph Synthesis ["Stage 6: Grounded Synthesis"]
        LLMReport["LLM Report Generator (Strict Source ID Constraints)"]
        Validator["Citation & Grounding Validator"]
    end

    UI -->|Submit Question| Input
    Input -->|POST /api/research| MW1
    MW1 --> MW2 --> MW3 --> Router --> Orchestrator

    Orchestrator --> Planner --> SubQ
    SubQ --> Tavily
    Tavily --> Scraper
    Scraper --> HTML & PDF
    HTML & PDF --> Chunker
    Chunker --> BM25
    BM25 --> EdgeEmbed
    EdgeEmbed --> PgVector
    PgVector --> MultiRetriever
    MultiRetriever --> ClaimExtract
    ClaimExtract --> Verifier
    Verifier --> ClaimDB
    Verifier --> LLMReport
    MultiRetriever --> LLMReport
    LLMReport --> Validator
    Validator --> Orchestrator
    Orchestrator -->|ResearchResponse JSON| Dossier
```

---

## 4. Research Pipeline

Every research request executed through `ResearchOrchestrator.run()` passes through an 11-stage lifecycle:

```
User Query
    │
    ▼
1. Session Initialization ───► Generates UUID, creates session in PostgreSQL with status 'pending'
    │
    ▼
2. Research Planning ────────► LLM creates ResearchPlan with 3-5 sub-questions & keyword queries
    │
    ▼
3. Concurrent Search ────────► Tavily API queries run via asyncio.Semaphore; URLs deduplicated
    │
    ▼
4. Source Persistence ───────► Saves unique sources to PostgreSQL 'sources' table
    │
    ▼
5. Web & PDF Extraction ─────► Concurrent fetch with BeautifulSoup (HTML) & PyPDF (PDFs)
    │
    ▼
6. Chunking & BM25 Filter ───► Sliding window chunks (1000/200); BM25 prunes to top 60 (max 6/doc)
    │
    ▼
7. Edge Embeddings ──────────► Supabase Edge Function (gte-small) embeds 60 chunks in batches of 4
    │
    ▼
8. Semantic Retrieval ───────► Multi-query pgvector cosine search across original & sub-queries
    │
    ▼
9. Claim Extraction ─────────► Extracts atomic claims from retrieved evidence chunks (E1..En)
    │
    ▼
10. Claim Verification ──────► Batched LLM verification (Supported, Partial, Unsupported, Insufficient)
    │
    ▼
11. Grounded Synthesis ──────► Synthesizes report; validates every [S#] citation against source index
```

### Stage Details

| Stage | Input | Processing | Output |
|---|---|---|---|
| **1. Session Init** | Natural language question | Allocates UUIDv4, writes row to `research_sessions` table with status `pending`. | `session_id` |
| **2. Planning** | Original question | LLM decomposes question into up to 5 orthogonal sub-questions with keyword search queries and justifications. | `ResearchPlan` |
| **3. Search** | Search queries from plan | Executes parallel search requests using `asyncio.Semaphore(min(len(sub_questions), 5))`. Deduplicates results by URL. | List of `SourceItem` |
| **4. Save Sources** | Raw source metadata | Persists sources in PostgreSQL with foreign key cascade to session. Sanitizes text. | `source_id_map` |
| **5. Extraction** | URLs & source metadata | Async HTTP GET with custom User-Agent. Content-type and magic-byte inspection routes to HTML parser or PDF extractor. Strips NUL (`\x00`) characters. Non-blocking error handling. | List of `Document` |
| **6. Chunking & Filter** | Extracted documents | Sliding window chunker (1000 chars, 200 overlap, break-boundary seeking). Pure-Python BM25 scores chunks across queries, selecting top 60 candidates with max 6 per document cap. | Top 60 `DocumentChunk` |
| **7. Embeddings** | Filtered chunk texts | Sends batches of 4 chunks to Supabase Edge Function via `asyncio.Semaphore(3)` with bounded concurrency. Vectors saved to `document_chunks` table. | 384-dim float vectors |
| **8. Retrieval** | Original Q + Sub-questions | Bounded concurrent pgvector cosine distance search (`<=>`). Deduplicates chunks across queries by maximum similarity. Top 15 chunks selected. | Top 15 `RetrievedChunk` |
| **9. Claim Extraction** | Top 15 chunks as evidence `E1`..`E15` | LLM extracts up to 15 discrete atomic factual statements, mapping each claim to candidate evidence IDs. | List of `ExtractedClaim` |
| **10. Claim Verification** | Extracted claims + evidence | Dual-path verifier (primary batched LLM call; fallback to semaphore-bounded individual calls). Classifies each claim into status with reason and confirmed evidence IDs. | List of `VerifiedClaim` |
| **11. Synthesis** | Verified claims + chunks | LLM synthesizes executive summary and sections. Strict validation verifies all `[S#]` citations exist in ingested sources. Status updated to `completed`. | `ResearchReport` |

---

## 5. Claim Verification Pipeline

Claim verification provides an explicit cross-examination stage between retrieval and synthesis:

```
Retrieved Evidence Chunks (Top 15)
           │
           ▼
  Assign Evidence Identifiers [E1, E2, ... E15]
           │
           ▼
  LLM Claim Extractor ────────► Extracts Atomic Claims [C1, C2, ...]
           │
           ▼
  Claim Verification Engine
  ┌────────────────────────────────────────────────────────┐
  │ Primary Path: Batched Verification                     │
  │   - Single prompt evaluates all C1..Cn in one LLM turn │
  │   - Strict JSON schema validation                      │
  │   - Validates all claim IDs returned without drops     │
  │                                                        │
  │ Fallback Path (On parse/network error):                │
  │   - Bounded concurrent individual verification        │
  │   - asyncio.Semaphore(concurrency=5)                   │
  │   - Individual errors degrade to INSUFFICIENT_EVIDENCE │
  └────────────────────────────────────────────────────────┘
           │
           ▼
  Deterministic Verification Status Assigned:
  ├── SUPPORTED: Directly confirmed by cited evidence
  ├── PARTIALLY_SUPPORTED: Partially confirmed; nuances qualified
  ├── UNSUPPORTED: Directly contradicted by evidence
  └── INSUFFICIENT_EVIDENCE: Ambiguous, missing, or inconclusive
           │
           ▼
  Persisted to 'claims' and 'claim_evidence' junction table
           │
           ▼
  Fed into Report Synthesis with Grounding Rules:
  - Supported claims -> Asserted as factual content
  - Partially supported -> Synthesized with explicit qualification
  - Unsupported -> Explicitly excluded or flagged as contradicted
  - Insufficient evidence -> Communicated with uncertainty
```

> **Conflicting & Inconclusive Evidence Handling**
> - When sources provide contradictory claims, the verifier assigns `PARTIALLY_SUPPORTED` or `UNSUPPORTED` with an explicit reason string detailing the conflict.
> - The report generator prompt enforces that unconfirmed statements cannot be stated as facts.
> - If evidence is missing, the system outputs `INSUFFICIENT_EVIDENCE` and communicates the limitation rather than generating ungrounded assertions.

---

## 6. Retrieval, Chunking & Embeddings

### Embedding Architecture (`gte-small` on Edge)
- **Model**: `gte-small` (General Text Embeddings, 384 dimensions, normalized).
- **Execution Environment**: Runs natively inside Supabase Edge Runtime using `Supabase.ai.Session('gte-small')`.
- **Platform Free Tier**: Executes via Supabase Edge Function invocations (included in Supabase's free-tier allocation of 500,000 invocations/month).
- **Batch Processing**: Texts are transmitted in batches of 4.
- **Bounded Concurrency**: Uses an `asyncio.Semaphore(3)` to prevent edge worker memory/CPU exhaust, while maintaining order alignment via `asyncio.gather()`.

### Sliding Window Chunking
Implemented in `TextChunker`:
- Default size: `CHUNK_SIZE = 1000` characters.
- Overlap: `CHUNK_OVERLAP = 200` characters.
- Natural boundary seeking: looks backwards within the overlap window for paragraph breaks (`\n\n`), newlines (`\n`), or spaces (` `) to avoid mid-sentence cuts.
- Whitespace & character normalization: strips NUL bytes (`\x00`), normalizes carriage returns, and collapses excessive whitespace.

### Pure-Python BM25 Candidate Filter
Implemented in `CandidateChunkFilter` (`k1=1.2, b=0.65`):
- **Why it is needed**: When multiple web pages and long-form PDF documents are fetched, raw chunk counts can easily reach 150–250 chunks. Sending all chunks to remote embedding functions increases latency, consumes edge memory, and writes redundant vectors.
- **How it works**: Scores every chunk against the composite query text:

  $$\text{Score} = \max(\text{BM25}_{\text{queries}}) + 0.2 \times \sum(\text{BM25}_{\text{queries}})$$

- **Diversity Capping**: Restricts candidate chunks before embedding to `CANDIDATE_CHUNKS_PRE_EMBED = 60` and enforces `CANDIDATE_CHUNKS_MAX_PER_DOC = 6`. This prevents a single lengthy document from dominating the candidate pool.

### pgvector Storage & Semantic Similarity
- Chunks are stored in PostgreSQL using the `vector(384)` column type with foreign keys cascading from `documents` and `research_sessions`.
- Similarity calculation:

  $$\text{Cosine Similarity} = 1 - (\text{embedding} \Leftrightarrow \text{query\_embedding})$$

- Multi-query retrieval runs vector queries for the main question and each sub-question concurrently via `asyncio.Semaphore(2)`. Chunks are merged across queries, keeping the maximum similarity score per chunk, sorted descending, and capped at `RETRIEVAL_MAX_TOTAL_CHUNKS = 15`.

---

## 7. Concurrency & Performance Engineering

The pipeline coordinates multiple external I/O boundaries using bounded concurrency:

```text
Sequential Operations (Unoptimized)                  Concurrent Operations (Implemented)
─────────────────────────────────────────────        ─────────────────────────────────────────────────
• Sequential search per sub-question (3-5x)   ───►   Bounded search via asyncio.gather() + Semaphore(5)
• Sequential webpage fetching                 ───►   Concurrent extraction via asyncio.gather()
• Raw chunks embedded without ceiling         ───►   BM25 filter caps candidates at 60 (max 6/doc)
• Embedding batches sent sequentially (c=1)   ───►   Bounded embedding concurrency via Semaphore(3)
• Sequential vector searches per sub-question ───►   Concurrent retrieval via Semaphore(2)
• Multiple individual claim verification calls───►   Primary single-request batched verification
```

### Key Optimizations

**1. Embedding Concurrency with Resource Bounds**
- Raw chunks are capped at 60 candidates before embedding, avoiding edge function payload bloat.
- Batches of 4 chunks are dispatched concurrently using `asyncio.Semaphore(3)`. This bounds edge worker concurrency to avoid memory and CPU limits on serverless runtimes while preserving vector-chunk alignment.

**2. Batched Claim Verification**
- In normal execution, all candidate claims are evaluated in a single LLM request (`_BatchClaimVerificationResponse`), reducing network roundtrips and token re-transmission.
- If response schema validation fails, the system automatically falls back to individual verification calls bounded by `asyncio.Semaphore(5)`.

**3. Concurrent Multi-Query Search & Retrieval**
- Search queries for all sub-questions execute concurrently using `asyncio.gather()`.
- Per-query vector retrievals execute concurrently using `asyncio.Semaphore(2)`.

**4. Connection Pooling & PgBouncer Compatibility**
- SQLAlchemy engine configured with `pool_size=5`, `max_overflow=10`, `pool_recycle=300s`, and `statement_cache_size: 0` in `connect_args` for full compatibility with Supabase transaction poolers on port `6543`.

---

## 8. Evaluation & Benchmark Suite

The repository includes a self-contained evaluation harness located in `backend/app/evaluation/`.

```bash
# Run benchmark across evaluation dataset
cd backend
python -m app.evaluation.run --limit 5
```

### Benchmark Dataset (`research_benchmark_v1.json`)
Contains 15 curated benchmark cases across multiple domains:
- **Science**: Net-gain fusion bottlenecks, mRNA vaccine immunology, cellular senescence mechanisms, solid-state battery electrolytes.
- **AI & ML**: Dense vs. MoE Transformer inference trade-offs, catastrophic forgetting in continual learning.
- **Engineering & Systems**: High-concurrency distributed consensus, Post-quantum cryptography.
- **General & Historical**: Younger Dryas impact hypothesis, Late Bronze Age collapse, Montreal Protocol enforcement.

Each benchmark case defines:
- The research question and category.
- `expected_dimensions`: Required topic dimensions with keyword match terms for deterministic evaluation.
- `expected_source_types`: Academic, technical documentation, or industry standards.
- `reference_facts`: Ground-truth factual statements.

### Measured Metrics

| Metric | Category | Description |
|---|---|---|
| **Planner Dimension Coverage** | Planning | Percentage of expected topic dimensions matched by generated sub-questions. |
| **Search Query Distinctness** | Planning | Ratio of unique search queries to total queries generated. |
| **Retrieval Hit Rate** | Retrieval | Percentage of queries (original + sub-questions) that successfully returned chunks. |
| **Average Similarity** | Retrieval | Mean cosine similarity score of top retrieved chunks. |
| **Evidence Coverage Ratio** | Retrieval | Unique source documents represented in retrieved chunks vs. total sources found. |
| **Supported Claim Rate** | Grounding | Percentage of extracted claims verified as `SUPPORTED`. |
| **Unsupported Claim Rate** | Grounding | Percentage of extracted claims verified as `UNSUPPORTED`. |
| **Traceability Integrity** | Traceability | Boolean verification confirming all cited evidence IDs map to valid chunk and document IDs. |
| **Citation Precision** | Citations | Ratio of valid cited sources to total citations in the report. |
| **Invalid Citation Count** | Citations | Count of hallucinated or out-of-index citation IDs (target: 0). |
| **End-to-End Latency** | Performance | Wall-clock latency per stage and overall session duration. |

---

## 9. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend Framework** | React 18.3 | User interface and reactive state management |
| **Build & Tooling** | Vite 6.0, TypeScript 5.6 | Development server, type checking, and production bundling |
| **Styling** | Tailwind CSS 3.4 | Responsive layout and dark-mode styling |
| **Icons** | Lucide React | Lightweight UI iconography |
| **Backend Framework** | FastAPI 0.115+ | Asynchronous REST API framework |
| **ASGI Server** | Uvicorn 0.30+ | Asynchronous server implementation |
| **Data Validation** | Pydantic v2.8+, Pydantic Settings | Request/response typing and environment validation |
| **Database & Vector** | Supabase PostgreSQL, `pgvector` 0.3+ | Relational persistence and 384-dim vector similarity search |
| **Database ORM** | SQLAlchemy 2.0+ (asyncio), asyncpg 0.30+ | Asynchronous PostgreSQL query execution and connection pooling |
| **Embeddings** | Supabase Edge Function (`Supabase.ai` `gte-small`) | 384-dimensional on-edge text embedding inference |
| **LLM Provider** | OpenAI-compatible Chat Completions API | Research planning, claim extraction, verification, and report synthesis |
| **Web Search** | Tavily Search API | Search engine optimized for LLM RAG pipelines |
| **HTML Extraction** | BeautifulSoup4 4.12+ | HTML DOM parsing, noise removal, and text cleaning |
| **PDF Extraction** | PyPDF 5.0+ | Binary PDF parsing and page-by-page text extraction |
| **HTTP Client** | HTTPX 0.27+ | Asynchronous HTTP client with connection pooling and timeouts |
| **Containerization** | Docker (python:3.11-slim) | Production container with unprivileged user |
| **Testing** | Pytest 8.3+, FastAPI TestClient | Unit, integration, and API test coverage |

---

## 10. Project Structure

<details>
<summary><strong>Click to expand full repository tree</strong></summary>

```text
AI-Research-Agent/
├── .env.example                         # Canonical environment variable reference
├── README.md                            # Comprehensive project documentation
├── docs/
│   └── deployment.md                    # Production deployment, PaaS setup & security architecture
├── supabase/
│   ├── config.toml                      # Supabase local CLI configuration
│   └── functions/
│       └── embed/                       # Supabase Edge Function for on-edge gte-small embeddings
│           └── index.ts                 # Deno handler using native Supabase.ai.Session('gte-small')
├── backend/
│   ├── Dockerfile                       # Multi-stage container, non-root user, dynamic $PORT
│   ├── requirements.txt                 # Backend Python dependencies
│   ├── app/
│   │   ├── main.py                      # FastAPI factory, middleware wiring, exception handlers
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── health.py            # Liveness (/health) and Readiness (/ready) probes
│   │   │       └── research.py          # Endpoints: /api/research, /plan, /retrieve, /verify
│   │   ├── core/
│   │   │   ├── config.py                # Pydantic Settings with production validation
│   │   │   ├── logging.py               # Structured logging with credential/URI masking
│   │   │   └── security.py              # RequestID, SecurityHeaders, InMemoryRateLimiter middlewares
│   │   ├── db/
│   │   │   ├── base.py                  # SQLAlchemy declarative Base
│   │   │   ├── session.py               # Async engine & sessionmaker with PgBouncer config
│   │   │   ├── models.py                # Models: ResearchSession, Source, Document, Chunk, Claim
│   │   │   └── migrations/              # Sequential SQL migration files
│   │   │       ├── 001_initial_schema.sql
│   │   │       ├── 002_pgvector_chunks.sql
│   │   │       ├── 003_claim_verification.sql
│   │   │       ├── 004_update_vector_dim_384.sql
│   │   │       └── 005_enable_rls.sql   # Enables RLS with default-deny on all tables
│   │   ├── schemas/                     # Typed Pydantic models for domain entities
│   │   │   ├── chunk.py
│   │   │   ├── claim.py
│   │   │   ├── document.py
│   │   │   ├── evidence.py
│   │   │   ├── planner.py
│   │   │   ├── report.py
│   │   │   ├── research.py
│   │   │   └── retrieval.py
│   │   ├── services/                    # Modular business logic services
│   │   │   ├── chunking/                # Sliding window text chunker
│   │   │   ├── claim/                   # Claim extractor and batched/semaphore verifiers
│   │   │   ├── embedding/               # Supabase Edge Function embedding provider
│   │   │   ├── extraction/              # BeautifulSoup HTML and PyPDF document extractor
│   │   │   ├── llm/                     # OpenAI-compatible report synthesis & citation logic
│   │   │   ├── persistence/             # SQLAlchemy asyncpg repository implementation
│   │   │   ├── pipeline/                # ResearchOrchestrator pipeline coordinator
│   │   │   ├── planner/                 # LLM research planner (sub-question generator)
│   │   │   ├── retrieval/               # pgvector retriever & pure-Python BM25 candidate filter
│   │   │   └── search/                  # Tavily API web search provider
│   │   └── evaluation/                  # Standardized benchmarking & metrics framework
│   │       ├── dataset.py               # Benchmark case loader
│   │       ├── metrics.py               # Deterministic calculation of quality metrics
│   │       ├── runner.py                # Headless pipeline evaluation runner
│   │       ├── run.py                   # CLI benchmark entrypoint
│   │       └── data/
│   │           └── research_benchmark_v1.json # 15-question evaluation dataset
│   └── tests/                           # 13 test modules covering all backend layers
│       ├── test_api.py                  # API endpoints, error responses, dependency injection
│       ├── test_chunking.py             # Boundary splits, normalization, overlap
│       ├── test_claim.py                # Extraction, batched verification, validation rules
│       ├── test_embedding.py            # Dimension checks, semaphore concurrency, batching
│       ├── test_evaluation.py           # Metric calculation tests, dataset validation
│       ├── test_extraction.py           # HTML cleaning, PDF parsing, NUL byte stripping
│       ├── test_llm.py                  # Report parsing, citation validation, fallbacks
│       ├── test_persistence.py          # Session lifecycle, pgvector search, cascade deletes
│       ├── test_planner.py              # Sub-question planning, limits validation
│       ├── test_production.py           # Production settings validation, rate limiting, security
│       ├── test_retrieval.py            # Cosine similarity queries, threshold filters
│       ├── test_search.py               # Tavily search provider, error handling
│       └── unit/
│           └── test_candidate_filter.py # BM25 scoring, diversity capping unit tests
└── frontend/
    ├── package.json                     # Frontend dependencies (React 18, Vite, Tailwind)
    ├── vite.config.ts                   # Vite configuration with /api proxy to backend
    ├── tailwind.config.js               # Dark-mode styling tokens
    └── src/
        ├── App.tsx                      # Main application shell and state orchestration
        ├── index.css                    # Tailwind directives and utility classes
        ├── types/api.ts                 # TypeScript interfaces matching backend Pydantic models
        ├── services/                    # API client with AbortController and custom error parsing
        │   ├── api.ts
        │   └── config.ts
        └── components/
            ├── common/                  # Badges, copy buttons, error alerts
            ├── layout/                  # Header and Footer
            └── research/                # Dossier, Claims, Sources, Plan, Report panels
```

</details>

---

## 11. API Documentation

### 1. Execute End-to-End Research
`POST /api/research`

Executes the full pipeline: planning, search, extraction, chunking, BM25 filtering, embedding, vector retrieval, claim extraction, verification, and report synthesis.

<details>
<summary><strong>Request / Response example (Illustrative)</strong></summary>

**Request:**
```http
POST /api/research HTTP/1.1
Content-Type: application/json

{
  "question": "What are the primary technical bottlenecks preventing commercial-scale net-gain fusion power?"
}
```

**Response (HTTP 200 OK — illustrative schema structure):**
```json
{
  "session_id": "a5c2d38e-49b1-4f12-9c31-7bfa5b8e9012",
  "question": "What are the primary technical bottlenecks preventing commercial-scale net-gain fusion power?",
  "status": "completed",
  "plan": {
    "original_question": "What are the primary technical bottlenecks preventing commercial-scale net-gain fusion power?",
    "sub_questions": [
      {
        "id": "q1",
        "question": "What are the supply and breeding challenges associated with tritium fuel in D-T fusion reactors?",
        "search_query": "tritium breeding ratio lithium blanket fusion bottlenecks",
        "reason": "Evaluates fuel self-sufficiency limits."
      }
    ]
  },
  "sources": [
    {
      "id": "S1",
      "title": "Tritium Breeding and Material Challenges in Magnetic Fusion",
      "url": "https://example.org/fusion-materials-review",
      "content": "Commercial D-T fusion requires self-sufficient tritium breeding...",
      "score": 0.89
    }
  ],
  "claims": [
    {
      "id": "C1",
      "claim": "Commercial D-T fusion requires breeding tritium using lithium blankets.",
      "status": "SUPPORTED",
      "reason": "Explicitly confirmed by evidence excerpt [E1].",
      "evidence_ids": ["E1"],
      "supporting_evidence_ids": ["E1"]
    }
  ],
  "report": {
    "title": "Technical Bottlenecks in Commercial Net-Gain Fusion Power",
    "summary": "Commercial fusion power faces three primary engineering barriers: tritium fuel self-sufficiency, neutron-induced material degradation, and plasma turbulence confinement [S1].",
    "sections": [
      {
        "heading": "Tritium Self-Sufficiency and Breeding",
        "content": "Deuterium-tritium (D-T) reactors require external startup tritium and continuous in-situ breeding through lithium blankets [S1]. Current breeding ratio projections remain tight.",
        "citations": ["S1"]
      }
    ],
    "sources": [
      {
        "id": "S1",
        "title": "Tritium Breeding and Material Challenges in Magnetic Fusion",
        "url": "https://example.org/fusion-materials-review"
      }
    ]
  },
  "message": "Planned 3 sub-questions, retrieved 5 source(s), verified 4 claim(s), and synthesized report."
}
```

</details>

### 2. Generate Research Plan
`POST /api/research/plan`

Decomposes a research question into structured sub-questions and search queries without running web search or retrieval.

<details>
<summary><strong>Request example</strong></summary>

```http
POST /api/research/plan HTTP/1.1
Content-Type: application/json

{
  "question": "How do solid-state electrolytes compare with liquid electrolytes in lithium batteries?"
}
```

</details>

### 3. Retrieve Relevant Chunks via Vector Similarity
`POST /api/research/retrieve`

Executes query embedding and retrieves top-$K$ document chunks using `pgvector` cosine similarity from a specified session.

<details>
<summary><strong>Request example</strong></summary>

```http
POST /api/research/retrieve HTTP/1.1
Content-Type: application/json

{
  "query": "dendrite formation in solid electrolytes",
  "session_id": "a5c2d38e-49b1-4f12-9c31-7bfa5b8e9012",
  "top_k": 5,
  "similarity_threshold": 0.65
}
```

</details>

### 4. Verify Claims
`POST /api/research/verify`

Evaluates candidate factual assertions against provided evidence excerpts.

<details>
<summary><strong>Request / Response example (Illustrative)</strong></summary>

**Request:**
```http
POST /api/research/verify HTTP/1.1
Content-Type: application/json

{
  "question": "Solid-state battery safety",
  "claims": [
    {
      "id": "C1",
      "claim": "Solid electrolytes eliminate all risk of thermal runaway.",
      "evidence_ids": ["E1"]
    }
  ],
  "evidence": [
    {
      "evidence_id": "E1",
      "title": "Solid State Battery Safety Review",
      "text": "While non-flammable solid electrolytes reduce fire hazards, short circuits from dendrites can still cause thermal breakdown under extreme conditions."
    }
  ]
}
```

**Response:**
```json
{
  "question": "Solid-state battery safety",
  "verified_claims": [
    {
      "id": "C1",
      "claim": "Solid electrolytes eliminate all risk of thermal runaway.",
      "status": "UNSUPPORTED",
      "reason": "Evidence explicitly states that short circuits from dendrites can still cause thermal breakdown.",
      "evidence_ids": ["E1"],
      "supporting_evidence_ids": []
    }
  ]
}
```

</details>

### 5. Health & Readiness Probes
- `GET /health`: Liveness probe for container orchestration. Returns `{"status": "ok"}` with HTTP 200 immediately without external dependencies.
- `GET /ready`: Readiness probe verifying PostgreSQL database connectivity. Executes `SELECT 1` with a bounded 3.0s timeout. Returns HTTP 200 on success, HTTP 503 on database disconnect.

---

## 12. Local Setup & Installation

### Prerequisites
- **Python**: 3.11 or 3.12
- **Node.js**: 18.x or 20.x (with `npm`)
- **Supabase Account**: For managed PostgreSQL with `pgvector` and Edge Functions
- **Tavily API Key**: [tavily.com](https://tavily.com/)
- **LLM API Key**: OpenAI key (`gpt-4o-mini`) or any OpenAI-compatible provider (e.g. local Ollama via `LLM_BASE_URL`)

### Step 1: Clone Repository
```bash
git clone https://github.com/[YOUR-USERNAME]/AI-Research-Agent.git
cd AI-Research-Agent
```

### Step 2: Database Setup (Supabase PostgreSQL)
1. Create a project at [supabase.com](https://supabase.com).
2. Open the **SQL Editor** in your Supabase dashboard.
3. Execute the migration scripts in `backend/app/db/migrations/` sequentially:
   - `001_initial_schema.sql` (Creates sessions, sources, documents tables)
   - `002_pgvector_chunks.sql` (Enables `vector` extension, creates `document_chunks`)
   - `003_claim_verification.sql` (Creates `claims` and `claim_evidence` tables)
   - `004_update_vector_dim_384.sql` (Sets vector column dimension to 384)
   - `005_enable_rls.sql` (Enables Row Level Security with default-deny on all tables)

### Step 3: Deploy Free On-Edge Embedding Function
Deploy the embedding Edge Function using the Supabase CLI:
```bash
# Login to Supabase CLI
npx supabase login

# Link your local project to your Supabase project ref
npx supabase link --project-ref [YOUR-PROJECT-REF]

# Deploy the embed edge function
npx supabase functions deploy embed --no-verify-jwt
```
Your embedding endpoint will be:
`https://[YOUR-PROJECT-REF].supabase.co/functions/v1/embed`

### Step 4: Configure Backend Environment
Create `.env` in the repository root from `.env.example`:
```bash
cp .env.example .env
```
Populate the configuration values in `.env`:
```ini
ENV=development
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:5173", "http://localhost:3000"]

# Web Search
TAVILY_API_KEY=tvly-...

# LLM Synthesis & Planning (OpenAI or compatible endpoint)
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Supabase Database & Edge Embeddings
SUPABASE_URL=https://[YOUR-PROJECT-REF].supabase.co
SUPABASE_ANON_KEY=[YOUR-SUPABASE-ANON-KEY]
SUPABASE_EMBEDDING_FUNCTION_URL=https://[YOUR-PROJECT-REF].supabase.co/functions/v1/embed
EMBEDDING_DIMENSIONS=384

# Database Connection (Transaction Pooler or Direct)
DATABASE_URL=postgresql+asyncpg://postgres.[YOUR-PROJECT-REF]:[YOUR-PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres
```

### Step 5: Backend Installation
```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Or on Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Frontend Installation
```bash
cd ../frontend

# Install npm dependencies
npm install

# Create local environment file
cp .env.example .env
```
*(In development, `VITE_API_BASE_URL` can remain empty to use the Vite dev proxy configured in `vite.config.ts`)*.

---

## 13. Environment Variables

<details>
<summary><strong>Click to expand full environment variable reference</strong></summary>

The table below lists all settings supported by `backend/app/core/config.py` and `.env.example`:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ENV` | Yes | `development` | Environment mode (`development` or `production`). Production activates strict configuration validation. |
| `API_HOST` | Yes | `0.0.0.0` | IP to bind the HTTP server. |
| `API_PORT` | Yes | `8000` | Port to bind Uvicorn. |
| `CORS_ORIGINS` | Yes | `["*"]` (dev only) | Allowed origin URLs (JSON array or comma-separated). Wildcard `*` is strictly forbidden in production. |
| `TAVILY_API_KEY` | Yes | `""` | API key for Tavily web search. |
| `SEARCH_PROVIDER` | No | `tavily` | Search provider adapter name. |
| `SEARCH_TIMEOUT_SECONDS` | No | `10.0` | Timeout per search query in seconds. |
| `SEARCH_MAX_RESULTS` | No | `5` | Maximum search results collected per sub-question. |
| `EXTRACTION_TIMEOUT_SECONDS` | No | `10.0` | Per-document HTTP fetch timeout in seconds. |
| `EXTRACTION_MAX_CHARS` | No | `50000` | Maximum character limit per extracted document. |
| `EXTRACTION_USER_AGENT` | No | *(Browser UA string)* | User-Agent header used for web scraping requests. |
| `LLM_API_KEY` | Yes | `""` | API key for OpenAI-compatible chat completions. |
| `LLM_BASE_URL` | No | `https://api.openai.com/v1` | Base URL for OpenAI-compatible chat completions API. |
| `LLM_MODEL` | No | `gpt-4o-mini` | LLM model identifier. |
| `LLM_TIMEOUT_SECONDS` | No | `30.0` | Timeout for report synthesis generation. |
| `LLM_MAX_CHARS_PER_DOC` | No | `4000` | Character ceiling per chunk/document provided to synthesis prompt. |
| `SUPABASE_URL` | Yes | `""` | Base Supabase project URL (`https://[ref].supabase.co`). |
| `SUPABASE_ANON_KEY` | No | `""` | Supabase public anonymous key (sent as Bearer token if present). |
| `SUPABASE_EMBEDDING_FUNCTION_URL` | Yes | `""` | Full URL to deployed `/functions/v1/embed` Edge Function. |
| `EMBEDDING_DIMENSIONS` | No | `384` | Vector dimensionality for `gte-small`. |
| `EMBEDDING_TIMEOUT_SECONDS` | No | `30.0` | Timeout for edge embedding batch requests. |
| `EMBEDDING_MAX_CONCURRENCY` | No | `3` | Maximum concurrent batch requests to edge embedding function. |
| `CHUNK_SIZE` | No | `1000` | Target characters per sliding text chunk. |
| `CHUNK_OVERLAP` | No | `200` | Overlap characters between consecutive chunks. |
| `CANDIDATE_CHUNKS_PRE_EMBED` | No | `60` | Maximum candidate chunks retained by BM25 before embedding. |
| `CANDIDATE_CHUNKS_MAX_PER_DOC` | No | `6` | Maximum candidate chunks allowed per source document. |
| `PLANNER_MAX_SUB_QUESTIONS` | No | `5` | Maximum sub-questions generated during planning. |
| `PLANNER_TIMEOUT_SECONDS` | No | `30.0` | Timeout for research planning LLM request. |
| `RETRIEVAL_TOP_K` | No | `5` | Top chunks retrieved per query vector search. |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | No | `None` | Optional minimum cosine similarity filter for chunk retrieval. |
| `RETRIEVAL_MAX_TOTAL_CHUNKS` | No | `15` | Total deduplicated chunks passed to claim extraction and synthesis. |
| `RETRIEVAL_CONCURRENCY` | No | `2` | Concurrent vector search queries against PostgreSQL. |
| `CLAIM_MAX_COUNT` | No | `15` | Maximum atomic claims extracted from evidence. |
| `CLAIM_EXTRACTION_TIMEOUT_SECONDS` | No | `30.0` | Timeout for claim extraction LLM request. |
| `CLAIM_VERIFICATION_TIMEOUT_SECONDS` | No | `30.0` | Timeout for claim verification LLM request. |
| `CLAIM_VERIFICATION_CONCURRENCY` | No | `5` | Concurrency limit for fallback individual claim verifications. |
| `CLAIM_VERIFICATION_BATCHED` | No | `true` | Enables single-prompt batched claim verification with fallback. |
| `DATABASE_URL` | Yes | `""` | Async PostgreSQL connection string (`postgresql+asyncpg://...`). |
| `DATABASE_POOL_SIZE` | No | `5` | SQLAlchemy base connection pool size. |
| `DATABASE_MAX_OVERFLOW` | No | `10` | Maximum connections allocated beyond base pool size. |
| `DATABASE_POOL_RECYCLE` | No | `300` | Connection recycling interval in seconds. |
| `DATABASE_TIMEOUT_SECONDS` | No | `10.0` | Database connection timeout in seconds. |
| `RATE_LIMIT_ENABLED` | No | `false` | Enables sliding-window rate limiting (set `true` in production). |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | No | `60` | Maximum requests per minute per IP. |
| `RATE_LIMIT_WINDOW_SECONDS` | No | `60` | Rate limiter tracking window in seconds. |

</details>

---

## 14. Running the Application

### Start the Backend
```bash
cd backend
# With virtual environment activated:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Backend API will be available at `http://127.0.0.1:8000`
- Swagger UI Documentation: `http://127.0.0.1:8000/docs`
- Health Probe: `http://127.0.0.1:8000/health`
- Readiness Probe: `http://127.0.0.1:8000/ready`

### Start the Frontend
```bash
cd frontend
npm run dev
```
Frontend development server will open at `http://localhost:5173`.

### Run via Docker
Build and run the production backend container:
```bash
cd backend
docker build -t ai-research-agent:latest .

docker run -d \
  --name research-agent-api \
  -p 8000:8000 \
  --env-file ../.env \
  ai-research-agent:latest
```

### Run Tests
```bash
cd backend
pytest tests/
```

---

## 15. Example Research Walkthrough

### Input Question
> *"What are the primary technical bottlenecks preventing commercial-scale net-gain fusion power?"*

### Pipeline Execution Trace

1. **Planning**: LLM decomposes the topic into 3 orthogonal sub-questions:
   - `q1`: Tritium fuel self-sufficiency and breeding blanket physics.
   - `q2`: 14 MeV neutron flux and first-wall materials degradation.
   - `q3`: High-beta plasma turbulence and magnet confinement limits.
2. **Search**: Tavily executes targeted queries in parallel. 5 unique sources collected; duplicates filtered.
3. **Extraction**: Async fetching parses HTML review articles and technical PDFs, cleaning scripts and extracting visible text.
4. **Chunking & BM25 Pre-Filtering**: Raw chunks generated via sliding window. BM25 scores all chunks against the combined queries, shortlisting candidate chunks up to the 60-chunk ceiling and applying the 6-chunk per-document cap.
5. **Edge Embedding**: Shortlisted chunks are embedded in batches of 4 via Supabase Edge Function (`gte-small`, 384 dimensions) using bounded concurrency. Vectors written to `document_chunks`.
6. **Semantic Retrieval**: Multi-query vector search runs across `q1`, `q2`, `q3`, and original query. Deduplicates chunks to top 15 by cosine similarity.
7. **Claim Extraction**: Extracts atomic factual statements mapped to evidence items `E1`..`E15`.
8. **Claim Verification**:

   | Claim | Result | Grounding Outcome |
   |---|---|---|
   | `C1`: *"Commercial D-T fusion requires breeding tritium using lithium blankets."* | **`SUPPORTED`** | Asserted as verified factual content. |
   | `C2`: *"14 MeV high-energy neutrons cause severe material displacement in the first wall."* | **`SUPPORTED`** | Asserted as verified factual content. |
   | `C3`: *"Net-gain fusion has already demonstrated sustained commercial grid electricity."* | **`UNSUPPORTED`** | Excluded or explicitly flagged as contradicted. |
   | `C4`: *"Commercial stellarators will be operational globally by 2026."* | **`INSUFFICIENT_EVIDENCE`** | Communicated with explicit uncertainty. |

9. **Grounded Synthesis**: LLM synthesizes report using claims guidance. Unsupported statements are excluded; valid statements cite `[S1]`, `[S2]`. Validated against source index.

---

## 16. Engineering Challenges & Solutions

### 1. Web Extraction Fragility & Binary PDFs
- **Problem**: Live web scraping encounters malformed HTML, SPA shells, binary PDFs, and NUL (`\x00`) characters that cause PostgreSQL driver errors.
- **Approach**: Multi-format extractor inspecting Content-Type, URL extensions, and magic bytes (`%PDF-`). HTML cleaned via BeautifulSoup with tag discard lists; PDFs parsed page-by-page with `pypdf`. Implemented `sanitize_document_text()` stripping NUL characters. All fetches run under `asyncio.gather(*tasks, return_exceptions=True)` with 10s timeouts.
- **Trade-off**: Heavy client-rendered SPAs without server-side rendering yield minimal text; handled gracefully by skipping sub-threshold documents (`MIN_TEXT_CHARS=40`).

### 2. Embedding Throughput vs. Edge Function Worker Limits
- **Problem**: When scraping multiple long web pages and PDFs, large chunk counts can cause edge runtime timeouts or memory limits if sent unconstrained.
- **Approach**: Two-tier architectural bounding:
  1. Pure-Python BM25 candidate filter pruning raw chunks to a ceiling of 60 candidates (capped at 6 per document) before remote embedding.
  2. Bounded concurrency using `asyncio.Semaphore(3)` sending batches of 4 to the Edge Function, preserving vector-to-chunk index ordering.
- **Trade-off**: Chunks ranked below the top 60 by BM25 are discarded prior to vector search. Document diversity capping ensures high-value chunks across all sources are preserved.

### 3. Claim Verification Latency & Reliability
- **Problem**: Verifying 10–15 claims individually creates multiple sequential network roundtrips, increasing latency.
- **Approach**: Implemented primary single-request batched verification with strict JSON schema validation, backed by an automated fallback to semaphore-bounded (`concurrency=5`) individual claim verification if parsing fails.
- **Trade-off**: Requires strict response validation to ensure all claim IDs are returned. The fallback guarantees system resilience if the batch payload is malformed.

### 4. Database Connection Drops & PgBouncer Compatibility
- **Problem**: Connecting to Supabase serverless PostgreSQL poolers caused prepared statement errors and stale connection drops.
- **Approach**: Configured SQLAlchemy with `pool_recycle=300`, `pool_pre_ping=True`, and disabled prepared statement caching (`statement_cache_size: 0` in `connect_args`).
- **Trade-off**: Disabling prepared statement caching adds minor parsing overhead in exchange for full compatibility with transaction poolers.

---

## 17. Key Technical Decisions

**Why RAG + A Dedicated Claim Verification Pass?**  
Ordinary RAG feeds retrieved excerpts directly into the synthesis prompt. The LLM can still misread, extrapolate, or hallucinate citations. Adding an explicit claim verification pass creates a cross-examination step: assertions are categorized into deterministic verification statuses before report generation begins.

**Why on-edge `gte-small` via Supabase Edge Runtime?**  
Using native `Supabase.ai` runs embeddings directly on Deno edge workers close to the database. It avoids third-party embedding API fees and uses available Supabase edge compute.

**Why a single-agent orchestrated pipeline instead of autonomous multi-agent frameworks?**  
Autonomous multi-agent frameworks introduce non-deterministic execution loops, higher latency, prompt bloat, and unpredictable token consumption. A deterministic, typed Python pipeline provides predictable latency, explicit error boundaries, testability, and production stability.

**Why PostgreSQL with pgvector?**  
Retains relational data (sessions, sources, documents, verified claims) alongside high-dimensional vector embeddings in a single ACID-compliant database. Eliminates synchronization complexity between external vector stores and relational databases.

---

## 18. Security & Reliability Measures

### Security Measures
- **Production Settings Validator**: Rejects placeholder credentials, unconfigured keys, and wildcard `*` CORS configurations when `ENV=production`.
- **Row Level Security (RLS)**: Enabled across all research tables with default-deny policies. *Note: The FastAPI backend connects directly via asyncpg using the privileged database role, which bypasses RLS in PostgreSQL; RLS specifically blocks unauthorized public data leakage through Supabase PostgREST endpoints (`/rest/v1/*`).*
- **Security Headers Middleware**: Injects `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy: strict-origin-when-cross-origin`.
- **Sensitive Data Masking**: Logging filter (`SensitiveDataFilter`) automatically scrubs database connection strings, passwords, and API keys from logs and error traces.
- **Docker Non-Root Execution**: Container runs under unprivileged `appuser` (UID 1000).

### Fault Tolerance & Operational Probes
- **Request ID Correlation**: Middleware assigns or propagates `X-Request-ID` using Python `contextvars` across asynchronous log messages and response headers.
- **Per-Stage Timeout Protections**: Search (10s), extraction (10s), planning (30s), embeddings (30s), claim extraction (30s), verification (30s), synthesis (30s).
- **Graceful Search & Scraping Degradation**: Individual search or extraction failures are logged as warnings and skipped; the pipeline continues as long as usable sources exist.
- **Database Readiness Probes**: `GET /ready` performs a bounded 3.0s `SELECT 1` ping without consuming third-party API quotas.

---

## 19. Cost & Free-Tier Architecture

The application architecture is designed around components that can be operated using available free-tier services or local runtimes:

| Component | Architecture Role | Free-Tier / Local Availability | Quota & Usage Considerations |
|---|---|---|---|
| **Supabase Database** | PostgreSQL storage & `pgvector` | Free tier includes 500 MB storage | Projects on free tier pause after 1 week of inactivity. |
| **Supabase Edge AI** | On-edge `gte-small` embeddings | Free tier includes 500,000 edge invocations / month | Subject to monthly edge runtime limits. BM25 pre-filtering keeps invocations low. |
| **Tavily Web Search** | Search engine API | Free tier provides 1,000 searches / month | 3–5 searches per session $\approx$ 200–300 full research runs per month. |
| **LLM Provider** | Planning, claims & synthesis | OpenAI-compatible endpoint | Compatible with low-cost models (`gpt-4o-mini`), trial credits, or free local models (e.g. Ollama). |
| **Hosting (Render / Vercel)** | Container & Static Frontend | Free web service & static tiers | Free compute instances may experience cold starts on inactivity. |

> [!NOTE]
> Free-tier allocations have usage quotas and are governed by third-party provider policies. The system's BM25 candidate filter, URL deduplication, and batched claim verification are specifically designed to minimize external API consumption.

---

## 20. Future Roadmap

**v1 (Current — Implemented)**
- [x] End-to-end research orchestration pipeline.
- [x] Multi-angle query decomposition.
- [x] Live web search and HTML/PDF document extraction.
- [x] Pure-Python BM25 pre-filtering with document diversity capping.
- [x] On-edge embeddings (`gte-small`) via Supabase Edge Functions.
- [x] Semantic retrieval with PostgreSQL and `pgvector`.
- [x] Atomic factual claim extraction.
- [x] Dedicated claim verification pass (batched + semaphore fallback).
- [x] Grounded report synthesis with strict citation validation.
- [x] Standardized 15-case evaluation benchmark suite.
- [x] Responsive React + Vite + TypeScript frontend.

**v2 (Planned Multi-Agent Architecture)**
- [ ] **Dynamic Research Planning Agent**: Evaluates intermediate retrieval gaps and dynamically formulates follow-up queries.
- [ ] **Source Credibility Evaluation Agent**: Implements domain reputation scoring and automated paywall/bot-block detection.
- [ ] **Adversarial Cross-Examination Agent**: Introduces a dedicated critic agent to challenge and stress-test claims before final verification.
- [ ] **Multi-Format Export Engine**: Native export of research dossiers to cited PDF, LaTeX, and Word formats.

---

## 21. Why This Project Matters

This project demonstrates core competencies in real-world AI and Backend Engineering:

1. **Information Retrieval Beyond Generic Tutorials**: Implements custom sliding-window chunking with boundary seeking, pure-Python BM25 candidate pre-filtering, document diversity capping, and multi-query vector retrieval.
2. **Mitigating LLM Hallucinations**: Demonstrates how to design an automated verification system that extracts claims, evaluates evidence support, and enforces strict citation integrity.
3. **Asynchronous Systems & Concurrency**: Leverages Python's `asyncio` primitives (`gather`, `Semaphore`, `contextvars`) to optimize multi-service I/O pipelines without triggering resource exhaustion.
4. **Defensive Production Engineering**: Implements health/readiness probes, sliding-window rate limiting, security headers, PostgreSQL Row Level Security, sensitive data log filtering, and container security.
5. **Empirical Evaluation**: Validates system accuracy and grounding against a standardized multi-domain benchmark dataset.

---

## 22. Interview Discussion Points

Technical questions this architecture is prepared to address:

1. **Why use a single-agent orchestrator rather than a multi-agent framework?**  
   *Predictable latency, lower token overhead, deterministic error boundaries, and simplified debugging.*
2. **Why implement BM25 candidate pre-filtering before vector embeddings?**  
   *Prunes raw chunks to a deterministic ceiling of 60 candidates (max 6 per document) before remote embedding, preventing edge worker memory exhaustion and single-source bias.*
3. **How does the system prevent hallucinated citations?**  
   *Ingested sources receive strict IDs (`S1`, `S2`). The synthesis schema requires section-level citations, and a post-generation validation gate rejects any citation not in the source map.*
4. **How are conflicting web sources handled?**  
   *Claims are evaluated as `PARTIALLY_SUPPORTED` or `UNSUPPORTED` with conflict explanations; grounding rules instruct the LLM to highlight disagreements.*
5. **Why run embeddings in a Supabase Edge Function instead of an external API?**  
   *Runs native on-edge `gte-small` within edge invocation quotas, reducing reliance on external token pricing and API keys.*
6. **How is database connection pooling handled with serverless PostgreSQL?**  
   *SQLAlchemy asyncpg pool recycling (300s), pre-ping validation, and disabled prepared statement caching for PgBouncer compatibility.*
7. **What happens when a scraped webpage contains binary PDF data or malformed HTML?**  
   *Content-type and magic-byte routing extracts text via PyPDF or BeautifulSoup; empty or unparseable documents are dropped gracefully without failing the pipeline.*
8. **How does the claim verifier balance latency against reliability?**  
   *Primary execution uses a single batched prompt; if validation fails, it falls back to semaphore-bounded individual verification calls.*

---

## 23. Resume Description

```text
AI Research Agent | FastAPI, Python, React, TypeScript, Supabase, pgvector, Docker
• Architected an asynchronous, evidence-grounded AI research pipeline that formulates multi-angle search plans, retrieves live web and PDF content, indexes vector embeddings, and synthesizes cited research reports.
• Engineered an on-edge embedding and retrieval pipeline using gte-small embeddings via Supabase Edge Functions and PostgreSQL pgvector cosine similarity search.
• Implemented a factual claim verification engine with batched LLM evaluation, strict citation validation, and automated fallback to semaphore-bounded concurrent verification.
• Designed a pure-Python BM25 lexical pre-filtering engine with document diversity capping, bounding candidates to a maximum of 60 chunks before remote embedding to eliminate single-source bias.
• Hardened backend for production with request ID correlation, security response headers, sliding-window rate limiting, sensitive data log scrubbing, and an automated 15-case evaluation benchmark suite.
```

---

<div align="center">

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

</div>
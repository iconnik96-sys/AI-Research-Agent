# AI Research Agent

## Goal

Build a production-quality AI Research Agent that can take a research question, search the web, collect and process sources, retrieve relevant evidence, reason over the evidence, verify claims, and generate a cited research report.

The project is intended as a serious AI Engineering portfolio project.

## Primary Objective

Build the smallest working version first, then progressively add:

1. Web research
2. Source extraction
3. Document chunking
4. Embeddings
5. Supabase pgvector retrieval
6. Research planning
7. Evidence management
8. Claim verification
9. Citation mapping
10. Evaluation
11. Production UI

Do NOT over-engineer the first version.

---

# Technology Stack

## Frontend

* React
* Tailwind CSS

## Backend

* Python
* FastAPI
* Pydantic
* Async Python where useful

## Database

* Supabase PostgreSQL
* pgvector

## AI

* LLM API
* Embedding API

## Search

* Web search API

## Processing

* HTML extraction
* PDF extraction
* Text chunking

## Infrastructure

* Docker
* GitHub

---

# Core Architecture

User
↓
React Frontend
↓
FastAPI Backend
↓
Research Planner
↓
Search Tools
↓
Source Extraction
↓
Document Processing
↓
Chunking
↓
Embeddings
↓
Supabase pgvector
↓
Retrieval + Reranking
↓
Evidence Store
↓
Claim Verification
↓
Report Generation
↓
Citations
↓
Frontend

---

# Development Strategy

Build in small vertical slices.

Never build the entire system in one task.

Every major feature must first have a minimal working implementation.

Prioritize:

1. Working functionality
2. Correct AI architecture
3. Evidence traceability
4. Testability
5. Performance
6. UI polish

Avoid unnecessary abstractions and dependencies.

---

# V1

The first version must support:

1. User enters a research question.
2. Backend receives the question.
3. Agent searches the web.
4. Agent collects several relevant sources.
5. Agent extracts source text.
6. Agent sends relevant information to the LLM.
7. Agent generates a structured research report.
8. Every important claim should have a source citation.
9. Frontend displays the report and sources.

Do NOT implement advanced autonomous planning, complex multi-agent systems, or sophisticated evaluation in V1.

---

# V2

Add:

* Supabase persistence
* research sessions
* sources
* documents
* chunks
* embeddings
* pgvector
* semantic retrieval

---

# V3

Add:

* research planner
* sub-question generation
* multiple search iterations
* retrieval/reranking
* evidence collection

---

# V4

Add:

* claims
* evidence mapping
* claim verification
* citation validation
* unsupported-claim detection

---

# V5

Add evaluation:

* retrieval evaluation
* citation correctness
* evidence support
* answer relevance
* latency
* token/API cost

Create a benchmark dataset of approximately 50–100 research questions.

---

# Engineering Rules

1. Keep frontend and backend separated.
2. Use environment variables for API keys.
3. Never hardcode secrets.
4. Use typed Pydantic models for backend APIs.
5. Keep AI orchestration logic separate from API routes.
6. Keep database access separate from business logic.
7. Make search providers replaceable.
8. Make LLM providers replaceable.
9. Store source URLs and metadata.
10. Every generated research claim should be traceable to evidence where possible.
11. Prefer deterministic, testable Python functions over unnecessary agent frameworks.
12. Avoid building multiple agents unless there is a clear engineering reason.
13. Do not add dependencies unless they solve a real problem.
14. Build and test each module before moving to the next.
15. Keep the initial implementation simple.

---

# Important AI Engineering Components

The project should demonstrate understanding of:

* LLM APIs
* embeddings
* vector search
* RAG
* chunking
* retrieval
* reranking
* prompt engineering
* tool calling
* research planning
* evidence grounding
* citation generation
* claim verification
* evaluation
* latency/cost optimization

---

# Antigravity Instructions

Before modifying the project:

1. Read brain.md.
2. Inspect only the files relevant to the current task.
3. Do not rewrite unrelated files.
4. Do not install unnecessary dependencies.
5. Complete one milestone/task at a time.
6. Run targeted tests after implementation.
7. Report:

   * files created
   * files modified
   * dependencies added
   * commands executed
   * tests/results
   * remaining issues

Do not redesign the architecture unless explicitly requested.

---

# Current Priority

CURRENT MILESTONE:

V1 — Minimal Working Research Agent

CURRENT TASK:

Create the project foundation and a minimal FastAPI backend that can accept a research question.

Do not implement RAG, pgvector, multi-agent orchestration, authentication, or advanced UI yet.

The next tasks will be provided separately.

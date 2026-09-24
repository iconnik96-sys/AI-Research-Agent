# Deployment & Production Architecture Guide

This document describes the production configuration, security architecture, deployment procedures, and operational guidelines for the **AI Research Agent** backend.

---

## 1. Architecture Overview

The system consists of:
- **FastAPI Backend**: Python 3.11/3.12 async application running in a Docker container.
- **Supabase PostgreSQL**: Managed PostgreSQL database with `pgvector` extension enabled.
- **External AI/Search Services**:
  - Web Search: Tavily API
  - LLM Synthesis: OpenAI-compatible API (`gpt-4o-mini`)
  - Embeddings: Supabase built-in AI inference (`gte-small`, 384 dimensions) via Supabase Edge Function

```text
[ Client (Browser / Next.js / React) ]
                   │
                   ▼ HTTPS (TLS)
┌────────────────────────────────────────────────────────┐
│  Reverse Proxy / Cloud Platform (Render / Railway)      │
│  - Edge TLS termination                                │
│  - Edge DDoS and network abuse protection              │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼ HTTP (Internal / Container)
┌────────────────────────────────────────────────────────┐
│  FastAPI Container (Docker: python:3.11-slim)          │
│                                                        │
│  Middlewares:                                          │
│  ├── RequestIDMiddleware (X-Request-ID correlation)   │
│  ├── SecurityHeadersMiddleware (nosniff, DENY, origin) │
│  ├── CORSMiddleware (Explicit allowed origins)         │
│  └── InMemoryRateLimiterMiddleware (Per-worker limiter)│
│                                                        │
│  Endpoints:                                            │
│  ├── GET  /health (Liveness probe, 200 OK)             │
│  ├── GET  /ready  (Readiness probe, SELECT 1 to DB)    │
│  └── POST /api/research (Core research pipeline)       │
└──────────────┬─────────────────────────┬───────────────┘
               │                         │
               ▼                         ▼
   [ Supabase PostgreSQL ]     [ External APIs ]
   - asyncpg connection pool   - Tavily Web Search
   - pgvector embeddings       - OpenAI LLM & Embeddings
```

---

## 2. Production Environment Variables

All secrets and environment-specific settings must be provided as environment variables to the deployment platform. **Never commit `.env` files into source control.**

| Variable | Required | Default | Description |
|---|---|---|---|
| `ENV` | Yes | `development` | Set to `production` in live environments. |
| `API_HOST` | Yes | `0.0.0.0` | IP to bind the HTTP server. |
| `API_PORT` | Yes | `8000` | Port to bind (or auto-assigned by `$PORT` on PaaS). |
| `CORS_ORIGINS` | Yes | `["*"]` (dev only) | Explicit list of trusted origins, e.g. `["https://research-agent.vercel.app"]`. **Wildcard `*` is strictly forbidden in production.** |
| `DATABASE_URL` | Yes | None | Supabase connection string: `postgresql+asyncpg://postgres.[ref]:[pass]@[host]:[port]/postgres`. |
| `DATABASE_POOL_SIZE` | No | `5` | Base connection pool size. |
| `DATABASE_MAX_OVERFLOW` | No | `10` | Maximum temporary connections above pool size. |
| `DATABASE_POOL_RECYCLE` | No | `300` | Recycles idle connections every 300s to avoid stale TCP drops. |
| `DATABASE_TIMEOUT_SECONDS`| No | `10.0` | Connection timeout in seconds. |
| `TAVILY_API_KEY` | Yes | None | Tavily API key for web search. |
| `LLM_API_KEY` | Yes | None | API key for LLM report synthesis. |
| `LLM_BASE_URL` | No | `https://api.openai.com/v1` | Base URL for OpenAI-compatible chat API. |
| `LLM_MODEL` | No | `gpt-4o-mini` | LLM model identifier. |
| `SUPABASE_URL` | No | None | Base URL for your Supabase project (e.g. `https://[ref].supabase.co`). |
| `SUPABASE_ANON_KEY` | No | None | Supabase public anon key (optional if Edge Function deployed with `--no-verify-jwt`). |
| `SUPABASE_EMBEDDING_FUNCTION_URL` | Yes | None | Full URL to the embed Edge Function (`https://[ref].supabase.co/functions/v1/embed`). |
| `EMBEDDING_DIMENSIONS` | No | `384` | Vector dimensionality for Supabase.ai `gte-small`. |
| `RATE_LIMIT_ENABLED` | No | `true` | Toggle in-memory rate limiting. |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | No | `10` | Max requests per minute per IP for research endpoints. |

---

## 3. Database Setup & Migrations

Before launching the backend, run the migration scripts in sequential order against your Supabase PostgreSQL database:

1. **`001_initial_schema.sql`**: Creates `research_sessions`, `sources`, and `documents` tables with foreign key cascades and indexes.
2. **`002_pgvector_chunks.sql`**: Enables the `vector` extension and creates the `document_chunks` table with `vector(384)` embeddings.
3. **`003_claim_verification.sql`**: Creates `claims` and `claim_evidence` tables for independent verification and evidence traceability.
4. **`004_update_vector_dim_384.sql`**: Updates existing `document_chunks` tables from `vector(1536)` to `vector(384)`.

> [!NOTE]
> If connecting to Supabase via the Transaction Pooler (port `6543`), the application automatically sets `statement_cache_size: 0` in `connect_args` for full PgBouncer compatibility.

---

## 4. Health & Readiness Endpoints

The backend provides two distinct health check endpoints:

### Liveness Probe (`GET /health`)
- **Purpose**: Checks if the container process is alive and responsive.
- **Behavior**: Returns `{"status": "ok"}` with HTTP 200 immediately.
- **Independence**: Never makes external database or API calls. If the container process is running, it returns 200.
- **Exempt from Rate Limiting**: Yes.

### Readiness Probe (`GET /ready`)
- **Purpose**: Checks if the backend is ready to accept user traffic by verifying database connectivity.
- **Behavior**: Executes a lightweight `SELECT 1` query against PostgreSQL with a bounded 3.0-second timeout.
  - Success: `HTTP 200` with `{"status": "ready", "database": "connected"}`.
  - Failure/Timeout: `HTTP 503` with `{"status": "unhealthy", "database": "disconnected"}`.
- **Independence**: Does **not** query Tavily or OpenAI APIs (avoiding unnecessary cost, latency, and third-party rate limits).
- **Exempt from Rate Limiting**: Yes.

---

## 5. Security & Rate Limiting Guidelines

### Security Headers
The backend automatically injects standard security response headers on all requests:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `X-Request-ID: <uuid>` (propagates incoming or generates a unique correlation ID)

### In-Memory Rate Limiting
- **Nature**: Provides lightweight, dependency-free, per-instance application-level abuse protection.
- **Mechanism**: A sliding-window tracking client IPs. Defaults to 10 requests per minute per IP for pipeline endpoints.
- **Exemptions**:
  - `GET /health` and `GET /ready` are never rate-limited.
  - CORS preflight `OPTIONS` requests are never rate-limited.
- **Behavior on Violation**: Returns `HTTP 429 Too Many Requests` with a `Retry-After: 60` response header.
- **Architectural Scope**:
  - **Per-process/worker**: Limits are stored in-memory within each Uvicorn worker process and are **not** shared across multiple workers or horizontal container replicas.
  - **Not DDoS Protection**: In-memory rate limiting protects upstream AI quotas from casual scraping or UI spam. True distributed denial-of-service (DDoS) protection **must** be provided by the edge/platform layer (e.g. Cloudflare, Render DDoS mitigation, AWS Shield).

---

## 6. Docker Container Deployment

### Local Docker Build & Run
```bash
# Navigate to backend directory
cd backend

# Build Docker image
docker build -t ai-research-agent-backend:latest .

# Run container with environment variables
docker run -d \
  --name research-agent \
  -p 8000:8000 \
  -e ENV=production \
  -e DATABASE_URL="postgresql+asyncpg://..." \
  -e TAVILY_API_KEY="tvly-..." \
  -e LLM_API_KEY="sk-..." \
  -e SUPABASE_EMBEDDING_FUNCTION_URL="https://[YOUR-PROJECT-REF].supabase.co/functions/v1/embed" \
  -e CORS_ORIGINS='["https://your-frontend.com"]' \
  ai-research-agent-backend:latest
```

---

## 7. Recommended Cloud Deployment (Render / Railway)

### Deploying to Render
1. Create a new **Web Service** in Render and connect your GitHub repository.
2. Set **Root Directory** to `backend`.
3. Select **Docker** as the Environment.
4. Set the following **Environment Variables**:
   - `ENV`: `production`
   - `DATABASE_URL`: Your Supabase connection string.
   - `TAVILY_API_KEY`: Your Tavily API key.
   - `LLM_API_KEY`: Your OpenAI/LLM API key.
   - `SUPABASE_EMBEDDING_FUNCTION_URL`: `https://[YOUR-PROJECT-REF].supabase.co/functions/v1/embed`
   - `CORS_ORIGINS`: `["https://your-frontend.vercel.app"]`
5. Configure Health Check:
   - **Health Check Path**: `/health`
6. Click **Deploy**.

-- Migration: 005_enable_rls.sql
-- Description: Enable Row Level Security (RLS) on all research tables to protect against unauthorized PostgREST/public access.
--
-- Security Design & Architecture:
-- 1. Direct Backend Access: The FastAPI backend connects directly to PostgreSQL via asyncpg using the 'postgres'
--    role (or 'service_role'). In PostgreSQL, roles with the BYPASSRLS attribute bypass RLS entirely.
--    Enabling RLS here does NOT restrict or affect the backend's direct database operations.
--
-- 2. PostgREST Default-Deny: Enabling RLS on each table activates a strict default-deny policy for non-bypass roles
--    ('anon' and 'authenticated'). Without permissive policies, any public/anon requests to Supabase PostgREST
--    endpoints (/rest/v1/*) are completely blocked, preventing public data leakage.
--
-- 3. No Fake Ownership: The current schema does not link research_sessions to Supabase auth.users(id).
--    Rather than inventing a mock user ownership system or using an insecure 'USING (true)' policy,
--    we enforce default-deny for client roles. All access must be routed through the FastAPI backend.
--
-- 4. Future Multi-Tenant Roadmap: If direct frontend-to-Supabase access is ever required:
--    a. Add a 'user_id UUID REFERENCES auth.users(id)' column to research_sessions.
--    b. Create scoped policies:
--       CREATE POLICY "Users can view own sessions" ON research_sessions
--           FOR SELECT TO authenticated USING (auth.uid() = user_id);

-- 1. Enable RLS on research_sessions
ALTER TABLE IF EXISTS research_sessions ENABLE ROW LEVEL SECURITY;

-- 2. Enable RLS on sources
ALTER TABLE IF EXISTS sources ENABLE ROW LEVEL SECURITY;

-- 3. Enable RLS on documents
ALTER TABLE IF EXISTS documents ENABLE ROW LEVEL SECURITY;

-- 4. Enable RLS on document_chunks
ALTER TABLE IF EXISTS document_chunks ENABLE ROW LEVEL SECURITY;

-- 5. Enable RLS on claims
ALTER TABLE IF EXISTS claims ENABLE ROW LEVEL SECURITY;

-- 6. Enable RLS on claim_evidence
ALTER TABLE IF EXISTS claim_evidence ENABLE ROW LEVEL SECURITY;

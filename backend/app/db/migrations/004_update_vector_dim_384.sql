-- Migration: 004_update_vector_dim_384.sql
-- Description: Update document_chunks table embedding column from vector(1536) to vector(384) for Supabase.ai gte-small embeddings.

-- 1. Truncate existing chunks (1536-dim vectors cannot be cast to 384 dimensions)
TRUNCATE TABLE document_chunks CASCADE;

-- 2. Alter column type to vector(384)
ALTER TABLE document_chunks 
    ALTER COLUMN embedding TYPE vector(384);

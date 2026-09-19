-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

-- Create agent_events table
CREATE TABLE agent_events (
    ts TIMESTAMPTZ NOT NULL,
    tenant_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    action TEXT NOT NULL,
    metadata JSONB
);

-- Convert agent_events into a TimescaleDB hypertable partitioned by time
SELECT create_hypertable('agent_events', 'ts');

-- Create intel table for hybrid RAG (semantic + relational)
CREATE TABLE intel (
    id SERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL,
    threat_level VARCHAR(20) NOT NULL,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content TEXT NOT NULL,
    embedding halfvec(384) -- Using halfvec for storage efficiency, assuming 384-dim embeddings (e.g., all-MiniLM-L6-v2)
);

-- Create HNSW index on the embedding column using cosine distance
CREATE INDEX intel_embedding_idx ON intel USING hnsw (embedding vector_cosine_ops);

-- Create partial index for exact matching and retrieve-then-rerank fallback
CREATE INDEX intel_tenant_threat_ts_idx ON intel (tenant_id, threat_level, ts);

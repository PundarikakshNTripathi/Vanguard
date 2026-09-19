import pytest
import asyncpg
import uuid
import os
from .rag import search_intel

# Using a standard pgvector connection string, assuming docker-compose environment
DB_URL = os.getenv("DATABASE_URL", "postgresql://vanguard:password@localhost:5432/vanguard_db")

@pytest.fixture
async def db_connection():
    # Setup connection
    conn = await asyncpg.connect(DB_URL)
    
    # Run migrations/setup for the test
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create the table if it doesn't exist (simulating schema.sql)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS intel (
            id SERIAL PRIMARY KEY,
            tenant_id UUID NOT NULL,
            threat_level VARCHAR(20) NOT NULL,
            ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            content TEXT NOT NULL,
            embedding halfvec(3)
        );
        CREATE INDEX IF NOT EXISTS intel_embedding_idx ON intel USING hnsw (embedding vector_cosine_ops);
    """)
    
    # Clean up table for this test
    await conn.execute("TRUNCATE intel;")
    
    yield conn
    
    # Teardown
    await conn.execute("TRUNCATE intel;")
    await conn.close()

@pytest.mark.asyncio
async def test_hybrid_rag_over_filtering(db_connection):
    conn = db_connection
    
    tenant_a = str(uuid.uuid4())
    tenant_b = str(uuid.uuid4())
    
    # We will insert a large number of rows for tenant_b to act as "distractors" for the HNSW index.
    # These distractors will be very close to the query vector, while the true matches for tenant_a 
    # will be further away. If iterative_scan is disabled, HNSW would fetch the nearest neighbors 
    # (which are mostly tenant_b), apply the tenant_a filter, and return 0 rows.
    # With iterative_scan enabled, it should correctly keep searching until it finds the tenant_a rows.
    
    # Insert distractors for tenant_b (close to query vector [1, 0, 0])
    await conn.executemany(
        "INSERT INTO intel (tenant_id, threat_level, content, embedding) VALUES ($1, $2, $3, $4)",
        [(tenant_b, "CRITICAL", f"Distractor {i}", "[0.9, 0.1, 0.0]") for i in range(100)]
    )
    
    # Insert true matches for tenant_a (further from query vector, e.g. [0.5, 0.5, 0.0])
    await conn.executemany(
        "INSERT INTO intel (tenant_id, threat_level, content, embedding) VALUES ($1, $2, $3, $4)",
        [(tenant_a, "CRITICAL", f"True match {i}", "[0.5, 0.5, 0.0]") for i in range(5)]
    )
    
    # Our query vector
    query_vec = [1.0, 0.0, 0.0]
    
    # Query for tenant_a
    results = await search_intel(
        conn=conn,
        tenant_id=tenant_a,
        threat_level="CRITICAL",
        days_lookback=30,
        query_embedding=query_vec,
        limit=5
    )
    
    # Verify we got the correct number of results despite the distractors
    assert len(results) == 5, "Hybrid RAG query failed to return all matching rows. Over-filtering may have occurred."
    
    # Verify all results belong to tenant_a
    for row in results:
        assert str(row['tenant_id']) == tenant_a
        assert row['threat_level'] == "CRITICAL"

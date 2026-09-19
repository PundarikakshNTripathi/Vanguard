import asyncpg
from typing import List, Tuple
from datetime import datetime, timedelta

async def search_intel(
    conn: asyncpg.Connection,
    tenant_id: str,
    threat_level: str,
    days_lookback: int,
    query_embedding: List[float],
    limit: int = 10
) -> List[asyncpg.Record]:
    """
    Executes a hybrid RAG query using pgvector's HNSW index alongside structured filters.

    WHY iterative_scan is used here:
    Combining an ANN vector index (HNSW) with a highly selective structured filter 
    (like `tenant_id` and `threat_level`) can lead to the "over-filtering" problem. 
    A bare HNSW search gathers a fixed set of nearest neighbors based purely on vector 
    distance, and ONLY THEN applies the structured filters. If the filter is very 
    selective, this fixed candidate set might contain very few (or zero) rows that 
    actually match the filter, even if perfectly matching rows exist deeper in the graph.
    
    Setting `hnsw.iterative_scan = 'relaxed_order'` fixes this. It forces the HNSW index 
    scan to iteratively expand its search graph until it finds enough matching rows that 
    satisfy BOTH the vector distance and the structured filters (up to a safety cap set by 
    `hnsw.max_scan_tuples`), ensuring we don't silently drop results.
    """
    
    # Configure session variables for this query to enable iterative scan safely
    await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order';")
    await conn.execute("SET LOCAL hnsw.max_scan_tuples = 20000;")
    
    # Execute the hybrid query
    query = """
        SELECT id, tenant_id, threat_level, ts, content
        FROM intel
        WHERE tenant_id = $1
          AND threat_level = $2
          AND ts > NOW() - $3::interval
        ORDER BY embedding <=> $4
        LIMIT $5;
    """
    
    interval_str = f"{days_lookback} days"
    
    # Convert query_embedding to the format expected by pgvector (string representation of array)
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
    
    results = await conn.fetch(query, tenant_id, threat_level, interval_str, embedding_str, limit)
    return results

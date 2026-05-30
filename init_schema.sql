-- Multilingual RAG schema for PostgreSQL (Neon-compatible)

CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    document_id SERIAL PRIMARY KEY,
    file_name VARCHAR(512) NOT NULL,
    language VARCHAR(16),
    chunk_count INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS query_logs (
    query_id SERIAL PRIMARY KEY,
    user_query TEXT NOT NULL,
    detected_language VARCHAR(16),
    response_time_ms INTEGER,
    retrieved_chunks INTEGER,
    answer TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

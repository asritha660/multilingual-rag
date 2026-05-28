"""
Database helper module for the Multilingual RAG Assistant.

Handles the PostgreSQL connection and provides functions to log
documents and queries into the tables defined in the project schema.
"""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Open a new connection to the PostgreSQL database using .env credentials."""
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME", "ragdb"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", ""),
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
    )


def log_document(file_name, language, chunk_count, metadata=None):
    """Insert a record into the documents table. Returns the new document_id."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (file_name, language, chunk_count, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING document_id;
                """,
                (file_name, language, chunk_count, json.dumps(metadata or {})),
            )
            doc_id = cur.fetchone()[0]
        conn.commit()
        return doc_id
    finally:
        conn.close()


def log_query(user_query, detected_language, response_time_ms,
              retrieved_chunks, answer):
    """Insert a record into the query_logs table. Returns the new query_id."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_logs
                    (user_query, detected_language, response_time_ms,
                     retrieved_chunks, answer)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING query_id;
                """,
                (user_query, detected_language, response_time_ms,
                 retrieved_chunks, answer),
            )
            query_id = cur.fetchone()[0]
        conn.commit()
        return query_id
    finally:
        conn.close()


def get_recent_queries(limit=20):
    """Return the most recent query log rows as a list of dicts."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT query_id, user_query, detected_language,
                       response_time_ms, retrieved_chunks, answer, created_at
                FROM query_logs
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            return cur.fetchall()
    finally:
        conn.close()

def create_user(username, hashed_password):
    """Insert a new user. Returns user_id, or None if username already taken."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (username, hashed_password)
                VALUES (%s, %s)
                RETURNING user_id;
                """,
                (username, hashed_password),
            )
            user_id = cur.fetchone()[0]
        conn.commit()
        return user_id
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return None  # username already exists
    finally:
        conn.close()


def get_user(username):
    """Look up a user by username. Returns dict with user_id, username, hashed_password, or None."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT user_id, username, hashed_password FROM users WHERE username = %s;",
                (username,),
            )
            return cur.fetchone()
    finally:
        conn.close()

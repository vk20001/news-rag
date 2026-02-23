"""
Metrics Logger — SQLite
========================

CONCEPT: Why Log Metrics?
-------------------------
Without logging, your pipeline is a black box. You run a query,
get an answer, and it's gone. With logging, every interaction is
recorded. This lets you:
1. Track faithfulness scores over time (is quality improving?)
2. Compare prompt versions (does v2 hallucinate less than v1?)
3. Monitor provider reliability (does Gemini fail more than Groq?)
4. Show a dashboard in Streamlit (visual proof of quality)
5. Feed into RAGAS evaluation in CI (automated quality checks)

CONCEPT: Why SQLite?
--------------------
- Zero setup — it's a file on disk, no server to run
- Built into Python — no pip install needed
- Perfect for single-user apps on 8GB RAM
- You already have Prometheus/Grafana on your CV — this proves
  you pick the right tool for the constraint, not the fanciest
"""

import sqlite3
import os
import json
from datetime import datetime, timezone


class MetricsLogger:

    def __init__(self, db_path: str = "data/metrics.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                num_chunks_retrieved INTEGER,
                best_chunk_distance REAL,
                faithfulness_score REAL,
                is_faithful INTEGER,
                is_refusal INTEGER,
                num_sentences INTEGER,
                num_flagged_sentences INTEGER,
                sources TEXT,
                latency_seconds REAL
            )
        """)

        conn.commit()
        conn.close()

    def log_query(
        self,
        query: str,
        answer: str,
        provider: str,
        model: str,
        prompt_version: str,
        chunks: list[dict],
        evaluation: dict,
        latency_seconds: float,
    ):
        """Log a complete query-answer cycle."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        sources = json.dumps([
            {"source": c["source"], "title": c["title"], "url": c["url"]}
            for c in chunks
        ])

        cursor.execute("""
            INSERT INTO query_logs (
                timestamp, query, answer, provider, model, prompt_version,
                num_chunks_retrieved, best_chunk_distance,
                faithfulness_score, is_faithful, is_refusal,
                num_sentences, num_flagged_sentences,
                sources, latency_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            query,
            answer,
            provider,
            model,
            prompt_version,
            len(chunks),
            chunks[0]["distance"] if chunks else None,
            evaluation.get("faithfulness_score"),
            int(evaluation.get("is_faithful", False)),
            int(evaluation.get("is_refusal", False)),
            evaluation.get("num_sentences", 0),
            len(evaluation.get("flagged_sentences", [])),
            sources,
            round(latency_seconds, 2),
        ))

        conn.commit()
        conn.close()

    def get_recent_logs(self, limit: int = 50) -> list[dict]:
        """Fetch recent query logs for dashboard display."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM query_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_summary_stats(self) -> dict:
        """Get aggregate stats for dashboard."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM query_logs")
        total = cursor.fetchone()[0]

        if total == 0:
            conn.close()
            return {"total_queries": 0}

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                AVG(faithfulness_score) as avg_faithfulness,
                SUM(is_faithful) as faithful_count,
                SUM(is_refusal) as refusal_count,
                AVG(latency_seconds) as avg_latency,
                AVG(best_chunk_distance) as avg_distance
            FROM query_logs
        """)

        row = cursor.fetchone()
        conn.close()

        return {
            "total_queries": row[0],
            "avg_faithfulness": round(row[1], 4) if row[1] else 0,
            "faithful_count": row[2],
            "refusal_count": row[3],
            "avg_latency": round(row[4], 2) if row[4] else 0,
            "avg_distance": round(row[5], 4) if row[5] else 0,
        }

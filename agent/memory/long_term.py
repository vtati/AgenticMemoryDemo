"""Long-term memory using SQLite for persistent storage across sessions.

Long-term memory stores user-specific information that persists across
different conversations/threads. This includes:
- User preferences (e.g., temperature units, communication style)
- Learned facts about the user (e.g., name, workplace, interests)
- Interaction patterns and history summaries

Key characteristics:
- User-scoped: Data is stored per user, not per thread
- Persistent: Survives application restarts
- Cross-session: Available in all conversations with the same user
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class LongTermMemory:
    """SQLite-based long-term memory for user profiles and facts."""

    def __init__(self, db_path: str | None = None):
        """Initialize long-term memory storage.

        Args:
            db_path: Path to SQLite database. Defaults to storage/long_term.db
        """
        if db_path is None:
            base_path = Path(__file__).parent.parent.parent
            db_path = str(base_path / "storage" / "long_term.db")

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            # User preferences table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, key)
                )
            """)

            # User facts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_user
                ON facts(user_id)
            """)

            conn.commit()

    def store_preference(self, user_id: str, key: str, value: str) -> None:
        """Store or update a user preference.

        Args:
            user_id: User identifier
            key: Preference key (e.g., 'temperature_unit', 'language')
            value: Preference value (e.g., 'celsius', 'english')
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO preferences (user_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
            """, (user_id, key, value, datetime.now()))
            conn.commit()

    def get_preference(self, user_id: str, key: str, default: str | None = None) -> str | None:
        """Get a user preference.

        Args:
            user_id: User identifier
            key: Preference key
            default: Default value if preference not found

        Returns:
            Preference value or default
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM preferences WHERE user_id = ? AND key = ?",
                (user_id, key)
            ).fetchone()
            return row["value"] if row else default

    def get_all_preferences(self, user_id: str) -> dict[str, str]:
        """Get all preferences for a user.

        Args:
            user_id: User identifier

        Returns:
            Dict of preference key-value pairs
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM preferences WHERE user_id = ?",
                (user_id,)
            ).fetchall()
            return {row["key"]: row["value"] for row in rows}

    def store_fact(
        self,
        user_id: str,
        fact_type: str,
        content: str,
        confidence: float = 1.0,
        source: str | None = None
    ) -> int:
        """Store a fact about the user.

        Args:
            user_id: User identifier
            fact_type: Type of fact (e.g., 'personal', 'work', 'interest')
            content: The fact content (e.g., 'Works at Acme Corp')
            confidence: Confidence score 0-1 (default 1.0)
            source: Where this fact was learned (e.g., 'user_stated')

        Returns:
            The inserted fact ID
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO facts (user_id, fact_type, content, confidence, source)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, fact_type, content, confidence, source))
            conn.commit()
            return cursor.lastrowid

    def get_facts(
        self,
        user_id: str,
        fact_type: str | None = None,
        limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get facts about a user.

        Args:
            user_id: User identifier
            fact_type: Optional filter by fact type
            limit: Maximum number of facts to return

        Returns:
            List of fact dictionaries
        """
        with self._get_connection() as conn:
            if fact_type:
                rows = conn.execute("""
                    SELECT id, fact_type, content, confidence, source, created_at
                    FROM facts
                    WHERE user_id = ? AND fact_type = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, fact_type, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT id, fact_type, content, confidence, source, created_at
                    FROM facts
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit)).fetchall()

            return [dict(row) for row in rows]

    def get_user_context(self, user_id: str) -> dict[str, Any]:
        """Get complete user context including preferences and facts.

        This is the main method used by the agent to load user context
        at the start of a conversation.

        Args:
            user_id: User identifier

        Returns:
            Dict with 'preferences' and 'facts' keys
        """
        preferences = self.get_all_preferences(user_id)
        facts = self.get_facts(user_id)

        return {
            "preferences": preferences,
            "facts": [f["content"] for f in facts],
            "fact_details": facts
        }

    def delete_fact(self, fact_id: int) -> bool:
        """Delete a specific fact by ID.

        Args:
            fact_id: The fact ID to delete

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_user_data(self, user_id: str) -> None:
        """Clear all data for a user (for privacy/GDPR).

        Args:
            user_id: User identifier
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM preferences WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))
            conn.commit()

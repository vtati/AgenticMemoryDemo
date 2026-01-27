"""Episodic memory using ChromaDB for storing and recalling past experiences.

Episodic memory stores past task executions as "episodes" that can be
semantically retrieved to guide similar future tasks. This enables:
- Learning from past successful task completions
- Providing few-shot examples based on similar past experiences
- Improving task execution over time

Key characteristics:
- Experience-based: Stores what was done, not just facts
- Semantic retrieval: Finds similar past tasks using vector similarity
- Action-oriented: Includes the steps/tools used to complete tasks
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings


class EpisodicMemory:
    """ChromaDB-based episodic memory for storing and recalling past experiences."""

    def __init__(self, persist_dir: str | None = None):
        """Initialize episodic memory storage.

        Args:
            persist_dir: Directory for ChromaDB persistence.
                        Defaults to storage/episodic
        """
        if persist_dir is None:
            base_path = Path(__file__).parent.parent.parent
            persist_dir = str(base_path / "storage" / "episodic")

        # Ensure directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create the episodes collection
        self.collection = self.client.get_or_create_collection(
            name="episodes",
            metadata={"description": "Past task executions and experiences"}
        )

    def store_episode(
        self,
        user_id: str,
        task: str,
        actions: list[str],
        outcome: str,
        success: bool = True,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Store a completed task as an episode.

        Args:
            user_id: User identifier
            task: Description of the task that was completed
            actions: List of actions/tools used (e.g., ['used calculator', 'wrote file'])
            outcome: The final result or output
            success: Whether the task was successful
            metadata: Optional additional metadata

        Returns:
            The episode ID
        """
        episode_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Combine task and actions for the embedding
        document = f"Task: {task}\nActions: {', '.join(actions)}\nOutcome: {outcome}"

        # Build metadata
        episode_metadata = {
            "user_id": user_id,
            "task": task,
            "actions": json.dumps(actions),
            "outcome": outcome,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }

        if metadata:
            # Add any additional metadata (must be primitive types)
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    episode_metadata[key] = value

        self.collection.add(
            ids=[episode_id],
            documents=[document],
            metadatas=[episode_metadata]
        )

        return episode_id

    def recall_similar(
        self,
        task: str,
        user_id: str | None = None,
        k: int = 3,
        min_similarity: float = 0.0
    ) -> list[dict[str, Any]]:
        """Find similar past tasks to guide current task execution.

        Args:
            task: Description of the current task
            user_id: Optional user ID to filter results
            k: Number of similar episodes to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of similar episodes with their details
        """
        # Build the query filter
        where_filter = None
        if user_id:
            where_filter = {"user_id": user_id}

        # Query for similar episodes
        results = self.collection.query(
            query_texts=[task],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        episodes = []

        if results and results["ids"] and results["ids"][0]:
            for i, episode_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                # Assuming L2 distance, approximate similarity
                similarity = 1 / (1 + distance)

                if similarity >= min_similarity:
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                    # Parse actions back from JSON
                    actions = []
                    if "actions" in metadata:
                        try:
                            actions = json.loads(metadata["actions"])
                        except json.JSONDecodeError:
                            actions = [metadata["actions"]]

                    episodes.append({
                        "episode_id": episode_id,
                        "task": metadata.get("task", ""),
                        "actions": actions,
                        "outcome": metadata.get("outcome", ""),
                        "success": metadata.get("success", True),
                        "timestamp": metadata.get("timestamp", ""),
                        "similarity": round(similarity, 3),
                        "document": results["documents"][0][i] if results["documents"] else ""
                    })

        return episodes

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        """Get a specific episode by ID.

        Args:
            episode_id: The episode ID

        Returns:
            Episode details or None if not found
        """
        results = self.collection.get(
            ids=[episode_id],
            include=["documents", "metadatas"]
        )

        if results and results["ids"]:
            metadata = results["metadatas"][0] if results["metadatas"] else {}

            # Parse actions
            actions = []
            if "actions" in metadata:
                try:
                    actions = json.loads(metadata["actions"])
                except json.JSONDecodeError:
                    actions = [metadata["actions"]]

            return {
                "episode_id": episode_id,
                "task": metadata.get("task", ""),
                "actions": actions,
                "outcome": metadata.get("outcome", ""),
                "success": metadata.get("success", True),
                "timestamp": metadata.get("timestamp", ""),
                "document": results["documents"][0] if results["documents"] else ""
            }

        return None

    def get_user_episodes(
        self,
        user_id: str,
        limit: int = 10,
        success_only: bool = False
    ) -> list[dict[str, Any]]:
        """Get all episodes for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of episodes to return
            success_only: If True, only return successful episodes

        Returns:
            List of episodes
        """
        where_filter = {"user_id": user_id}
        if success_only:
            where_filter["success"] = True

        results = self.collection.get(
            where=where_filter,
            limit=limit,
            include=["documents", "metadatas"]
        )

        episodes = []
        if results and results["ids"]:
            for i, episode_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}

                actions = []
                if "actions" in metadata:
                    try:
                        actions = json.loads(metadata["actions"])
                    except json.JSONDecodeError:
                        actions = [metadata["actions"]]

                episodes.append({
                    "episode_id": episode_id,
                    "task": metadata.get("task", ""),
                    "actions": actions,
                    "outcome": metadata.get("outcome", ""),
                    "success": metadata.get("success", True),
                    "timestamp": metadata.get("timestamp", "")
                })

        # Sort by timestamp descending
        episodes.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return episodes[:limit]

    def delete_episode(self, episode_id: str) -> bool:
        """Delete a specific episode.

        Args:
            episode_id: The episode ID to delete

        Returns:
            True if deleted
        """
        try:
            self.collection.delete(ids=[episode_id])
            return True
        except Exception:
            return False

    def clear_user_episodes(self, user_id: str) -> int:
        """Clear all episodes for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of episodes deleted
        """
        # Get all episode IDs for the user
        results = self.collection.get(
            where={"user_id": user_id},
            include=[]
        )

        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])

        return 0

    def get_stats(self, user_id: str | None = None) -> dict[str, Any]:
        """Get statistics about stored episodes.

        Args:
            user_id: Optional user ID to filter stats

        Returns:
            Statistics dictionary
        """
        if user_id:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
        else:
            results = self.collection.get(include=["metadatas"])

        total = len(results["ids"]) if results["ids"] else 0
        successful = 0

        if results["metadatas"]:
            successful = sum(1 for m in results["metadatas"] if m.get("success", True))

        return {
            "total_episodes": total,
            "successful_episodes": successful,
            "failed_episodes": total - successful,
            "success_rate": successful / total if total > 0 else 0
        }

"""Short-term memory using LangGraph's built-in checkpointer.

Short-term memory maintains conversation history within a single thread/session.
It is automatically managed by LangGraph's checkpointing system.

Key characteristics:
- Thread-scoped: Each thread_id has its own conversation history
- Automatic: Messages are automatically persisted by the checkpointer
- Session-based: Cleared when starting a new thread
"""

from langgraph.checkpoint.memory import MemorySaver


# Singleton checkpointer instance for the application
_checkpointer: MemorySaver | None = None


def get_checkpointer() -> MemorySaver:
    """Get or create the memory checkpointer for short-term memory.

    The checkpointer automatically manages conversation history for each thread.
    Use the thread_id in the config to maintain separate conversations:

    Example:
        config = {"configurable": {"thread_id": "user_123_session_1"}}
        result = await graph.ainvoke(input_state, config)

    Returns:
        MemorySaver instance for checkpointing
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def create_thread_config(user_id: str, session_id: str | None = None) -> dict:
    """Create a config dict with thread_id for checkpointing.

    Args:
        user_id: The user identifier
        session_id: Optional session identifier (defaults to timestamp-based)

    Returns:
        Config dict with thread_id for use with graph.invoke()
    """
    import time

    if session_id is None:
        session_id = str(int(time.time()))

    thread_id = f"{user_id}_{session_id}"
    return {"configurable": {"thread_id": thread_id}}

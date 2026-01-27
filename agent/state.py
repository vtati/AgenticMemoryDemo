"""Agent state schema for the LangGraph workflow.

This defines the state structure that flows through the agent graph,
including conversation messages and all three memory types.
"""

from typing import Annotated, Any, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State schema for the agentic AI workflow.

    This state is passed through all nodes in the graph and maintains:
    - Short-term memory: Conversation messages (via add_messages reducer)
    - Long-term memory: User preferences and facts loaded at start
    - Episodic memory: Similar past experiences for few-shot guidance
    - Task tracking: Current task and actions taken
    """

    # ===== Short-term Memory =====
    # Conversation history managed by LangGraph's message reducer
    # New messages are automatically appended to existing ones
    messages: Annotated[list, add_messages]

    # ===== Long-term Memory =====
    # User identifier (persists across sessions)
    user_id: str

    # User preferences loaded from SQLite (e.g., {"temperature_unit": "celsius"})
    user_preferences: dict[str, str]

    # Known facts about the user (e.g., ["Works at Acme Corp", "Likes hiking"])
    known_facts: list[str]

    # ===== Episodic Memory =====
    # Similar past experiences retrieved from ChromaDB
    similar_episodes: list[dict[str, Any]]

    # ===== Current Task Tracking =====
    # The current task being worked on
    current_task: str

    # List of actions/tools used during task execution
    task_actions: list[str]

    # The final outcome/result of the task
    task_outcome: str

    # Whether to store this as an episode (set after task completion)
    should_store_episode: bool


def create_initial_state(
    user_id: str,
    task: str,
    user_message: str
) -> dict[str, Any]:
    """Create the initial state for a new agent invocation.

    Args:
        user_id: The user identifier
        task: The current task description
        user_message: The user's message to start the conversation

    Returns:
        Initial state dictionary (partial - memory will be loaded by nodes)
    """
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=user_message)],
        "user_id": user_id,
        "user_preferences": {},
        "known_facts": [],
        "similar_episodes": [],
        "current_task": task,
        "task_actions": [],
        "task_outcome": "",
        "should_store_episode": False,
    }

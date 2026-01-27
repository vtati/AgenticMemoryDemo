"""LangGraph agent graph definition.

This module defines the main agent workflow graph that orchestrates
memory loading, reasoning, tool execution, and episode storage.
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    load_user_context,
    retrieve_episodes,
    call_agent,
    execute_tools,
    store_episode,
    route_agent,
)
from .memory import get_checkpointer


def create_agent_graph() -> StateGraph:
    """Create and compile the agent graph.

    The graph flow:
    1. load_context: Load user preferences/facts from long-term memory
    2. retrieve_episodes: Find similar past experiences
    3. agent: Claude reasoning with memory context
    4. [conditional] tools -> agent (loop) OR store/end

    Returns:
        Compiled StateGraph with checkpointing enabled
    """
    # Create the graph with our state schema
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("load_context", load_user_context)
    graph.add_node("retrieve_episodes", retrieve_episodes)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", execute_tools)
    graph.add_node("store", store_episode)

    # Define the flow
    # Start -> Load Context -> Retrieve Episodes -> Agent
    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "retrieve_episodes")
    graph.add_edge("retrieve_episodes", "agent")

    # Agent -> Conditional routing
    graph.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "store": "store",
            "end": END
        }
    )

    # Tools -> Back to Agent (the tool call loop)
    graph.add_edge("tools", "agent")

    # Store -> End
    graph.add_edge("store", END)

    # Compile with checkpointing for short-term memory
    checkpointer = get_checkpointer()
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled


def get_graph_visualization() -> str:
    """Get ASCII visualization of the graph structure.

    Returns:
        ASCII art representation of the graph
    """
    return """
    ┌─────────────────────────────────────────────────────────────┐
    │                         START                                │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    load_context                              │
    │         (Load preferences & facts from SQLite)               │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   retrieve_episodes                          │
    │         (Find similar past tasks from ChromaDB)              │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                        agent                                 │
    │              (Claude reasoning with tools)                   │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
              ┌─────────┐   ┌─────────┐   ┌─────────┐
              │  tools  │   │  store  │   │   END   │
              └────┬────┘   └────┬────┘   └─────────┘
                   │             │
                   │             ▼
                   │        ┌─────────┐
                   │        │   END   │
                   │        └─────────┘
                   │
                   └──────────────┐
                                  │
                                  ▼
                            (back to agent)
    """

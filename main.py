#!/usr/bin/env python3
"""Main CLI entry point for the Agentic AI with Memory Demo.

This provides an interactive command-line interface to interact with
the LangGraph agent that uses tools and demonstrates three memory types:
- Short-term: Conversation history within a session
- Long-term: User preferences/facts persisted across sessions
- Episodic: Past experiences recalled for similar tasks
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from agent import create_agent_graph, AgentState
from agent.state import create_initial_state
from agent.memory import LongTermMemory, EpisodicMemory


# Load environment variables
load_dotenv()


def print_banner():
    """Print the welcome banner."""
    print("\n" + "=" * 60)
    print("   Agentic AI with LangGraph, MCP Tools & Memory Systems")
    print("=" * 60)
    print()
    print("This demo showcases three types of memory:")
    print("  📝 Short-term: Conversation history (within session)")
    print("  💾 Long-term:  User preferences & facts (across sessions)")
    print("  🧠 Episodic:   Past experiences (for similar tasks)")
    print()
    print("Available tools:")
    print("  • calculator   - Math operations")
    print("  • read_file    - Read files from workspace")
    print("  • write_file   - Write files to workspace")
    print("  • list_files   - List workspace files")
    print("  • get_weather  - Get weather for a city")
    print()
    print("Commands:")
    print("  /newthread   - Start a new conversation thread")
    print("  /showmemory  - Show stored memories for current user")
    print("  /clearmemory - Clear all memories for current user")
    print("  /help        - Show this help message")
    print("  /quit        - Exit the application")
    print()
    print("=" * 60 + "\n")


def show_memory(user_id: str):
    """Display all stored memories for a user."""
    print("\n" + "-" * 40)
    print(f"Memory State for User: {user_id}")
    print("-" * 40)

    # Long-term memory
    long_term = LongTermMemory()
    context = long_term.get_user_context(user_id)

    print("\n📝 Long-term Memory (SQLite):")
    print("  Preferences:")
    if context["preferences"]:
        for key, value in context["preferences"].items():
            print(f"    • {key}: {value}")
    else:
        print("    (none)")

    print("  Facts:")
    if context["facts"]:
        for fact in context["facts"][:10]:
            print(f"    • {fact}")
    else:
        print("    (none)")

    # Episodic memory
    episodic = EpisodicMemory()
    stats = episodic.get_stats(user_id)
    episodes = episodic.get_user_episodes(user_id, limit=5)

    print(f"\n🧠 Episodic Memory (ChromaDB):")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Success rate: {stats['success_rate']:.0%}")

    if episodes:
        print("  Recent episodes:")
        for ep in episodes[:5]:
            print(f"    • Task: {ep['task'][:50]}...")
            if ep['actions']:
                print(f"      Actions: {', '.join(ep['actions'][:3])}")
    else:
        print("  (no episodes)")

    print("-" * 40 + "\n")


def clear_memory(user_id: str):
    """Clear all memories for a user."""
    print(f"\nClearing all memories for user: {user_id}")

    # Clear long-term memory
    long_term = LongTermMemory()
    long_term.clear_user_data(user_id)
    print("  ✓ Long-term memory cleared")

    # Clear episodic memory
    episodic = EpisodicMemory()
    count = episodic.clear_user_episodes(user_id)
    print(f"  ✓ Episodic memory cleared ({count} episodes)")

    print("Done!\n")


async def run_agent(
    graph,
    user_id: str,
    thread_id: str,
    user_input: str,
    store_episode: bool = False
) -> str:
    """Run the agent with the given input.

    Args:
        graph: Compiled LangGraph
        user_id: User identifier
        thread_id: Thread identifier for short-term memory
        user_input: User's message
        store_episode: Whether to store this as an episode

    Returns:
        The agent's response text
    """
    # Create initial state
    initial_state = create_initial_state(
        user_id=user_id,
        task=user_input,
        user_message=user_input
    )

    # Set episode storage flag
    initial_state["should_store_episode"] = store_episode

    # Config with thread_id for checkpointing
    config = {"configurable": {"thread_id": thread_id}}

    # Run the graph
    result = await graph.ainvoke(initial_state, config)

    # Extract the final response
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            # Handle both string and list content
            if isinstance(msg.content, str):
                return msg.content
            elif isinstance(msg.content, list):
                # Extract text from content blocks
                texts = []
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        texts.append(block)
                return "\n".join(texts)

    return "I completed the task but have nothing to add."


async def main():
    """Main entry point."""
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set!")
        print("Please set it in your .env file or export it:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    print_banner()

    # Initialize the graph
    print("Initializing agent graph...")
    graph = create_agent_graph()
    print("Agent ready!\n")

    # User and thread management
    user_id = "demo_user"
    thread_id = f"{user_id}_{int(time.time())}"
    interaction_count = 0

    print(f"User ID: {user_id}")
    print(f"Thread ID: {thread_id}")
    print("\nType your message and press Enter. Use /help for commands.\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()

                if cmd in ["/quit", "/exit", "/q"]:
                    print("\nGoodbye! Your memories are saved.\n")
                    break

                elif cmd == "/newthread":
                    thread_id = f"{user_id}_{int(time.time())}"
                    interaction_count = 0
                    print(f"\n✓ Started new thread: {thread_id}")
                    print("(Long-term and episodic memories persist)\n")
                    continue

                elif cmd == "/showmemory":
                    show_memory(user_id)
                    continue

                elif cmd == "/clearmemory":
                    confirm = input("Are you sure? This cannot be undone. (yes/no): ")
                    if confirm.lower() == "yes":
                        clear_memory(user_id)
                    else:
                        print("Cancelled.\n")
                    continue

                elif cmd == "/help":
                    print_banner()
                    continue

                else:
                    print(f"Unknown command: {user_input}")
                    print("Type /help for available commands.\n")
                    continue

            # Run the agent
            interaction_count += 1
            print()  # Blank line before agent output

            # Store as episode every few interactions
            store_episode = interaction_count % 3 == 0

            try:
                response = await run_agent(
                    graph=graph,
                    user_id=user_id,
                    thread_id=thread_id,
                    user_input=user_input,
                    store_episode=store_episode
                )

                print(f"Agent: {response}\n")

            except Exception as e:
                print(f"Error: {e}\n")
                import traceback
                traceback.print_exc()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Your memories are saved.")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())

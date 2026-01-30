"""Graph nodes for the LangGraph agent workflow.

Each node is a function that takes the current state and returns
updates to be merged into the state.
"""

import json
import re
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage

from .state import AgentState
from .memory import LongTermMemory, EpisodicMemory


# Initialize memory systems (singletons)
_long_term_memory: LongTermMemory | None = None
_episodic_memory: EpisodicMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    """Get or create the long-term memory instance."""
    global _long_term_memory
    if _long_term_memory is None:
        _long_term_memory = LongTermMemory()
    return _long_term_memory


def get_episodic_memory() -> EpisodicMemory:
    """Get or create the episodic memory instance."""
    global _episodic_memory
    if _episodic_memory is None:
        _episodic_memory = EpisodicMemory()
    return _episodic_memory


# Initialize the Claude model
def get_model() -> ChatAnthropic:
    """Get the Claude model instance."""
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.7,
        max_tokens=4096,
    )


# Define tools for the agent
TOOLS = [
    {
        "name": "calculator",
        "description": "Perform basic arithmetic operations (add, subtract, multiply, divide) on two numbers",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {"type": "number", "description": "The first operand"},
                "b": {"type": "number", "description": "The second operand"}
            },
            "required": ["operation", "a", "b"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file from the workspace directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file to read (relative to workspace)"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the workspace directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file to write (relative to workspace)"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "list_files",
        "description": "List all files in the workspace directory",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns temperature (Fahrenheit and Celsius), condition, and humidity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to get weather for"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "store_user_preference",
        "description": "Store a user preference for future conversations (e.g., temperature unit, language)",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The preference key (e.g., 'temperature_unit', 'name')"
                },
                "value": {
                    "type": "string",
                    "description": "The preference value (e.g., 'celsius', 'Alex')"
                }
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "store_user_fact",
        "description": "Store a fact about the user for future reference (e.g., workplace, interests)",
        "input_schema": {
            "type": "object",
            "properties": {
                "fact_type": {
                    "type": "string",
                    "description": "Category of fact (e.g., 'personal', 'work', 'interest')"
                },
                "content": {
                    "type": "string",
                    "description": "The fact to store (e.g., 'Works at Acme Corp')"
                }
            },
            "required": ["fact_type", "content"]
        }
    }
]


def build_system_prompt(state: AgentState) -> str:
    """Build the system prompt including memory context.

    Args:
        state: Current agent state with memory loaded

    Returns:
        System prompt string
    """
    prompt_parts = [
        "You are a helpful AI assistant with access to tools and memory capabilities.",
        "",
        "## Your Capabilities",
        "- Calculator for math operations",
        "- File operations (read, write, list) in the workspace",
        "- Weather lookup for cities",
        "- Store user preferences and facts for future conversations",
        "",
        "## Guidelines",
        "- Use tools when needed to complete tasks",
        "- Store important user information using store_user_preference or store_user_fact",
        "- Reference past experiences when relevant",
        "- Be concise but helpful",
    ]

    # Add user preferences if available
    if state.get("user_preferences"):
        prompt_parts.append("")
        prompt_parts.append("## User Preferences (from long-term memory)")
        for key, value in state["user_preferences"].items():
            prompt_parts.append(f"- {key}: {value}")

    # Add known facts if available
    if state.get("known_facts"):
        prompt_parts.append("")
        prompt_parts.append("## Known Facts About User (from long-term memory)")
        for fact in state["known_facts"][:10]:  # Limit to 10 facts
            prompt_parts.append(f"- {fact}")

    # Add similar past experiences if available
    if state.get("similar_episodes"):
        prompt_parts.append("")
        prompt_parts.append("## Similar Past Experiences (from episodic memory)")
        prompt_parts.append("Use these as guidance for how to approach similar tasks:")
        for episode in state["similar_episodes"][:3]:  # Limit to 3 episodes
            prompt_parts.append(f"- Task: {episode.get('task', 'Unknown')}")
            actions = episode.get('actions', [])
            if actions:
                prompt_parts.append(f"  Actions taken: {', '.join(actions)}")
            prompt_parts.append(f"  Outcome: {episode.get('outcome', 'Unknown')}")

    return "\n".join(prompt_parts)


# ===== Node Functions =====

async def load_user_context(state: AgentState) -> dict[str, Any]:
    """Node: Load user context from long-term memory.

    This runs at the start of each conversation to load
    the user's preferences and known facts.
    """
    print("[Memory] Loading user context from long-term memory...")

    memory = get_long_term_memory()
    user_context = memory.get_user_context(state["user_id"])

    preferences = user_context.get("preferences", {})
    facts = user_context.get("facts", [])

    print(f"[Memory] Loaded {len(preferences)} preferences, {len(facts)} facts")

    return {
        "user_preferences": preferences,
        "known_facts": facts
    }


async def retrieve_episodes(state: AgentState) -> dict[str, Any]:
    """Node: Retrieve similar past experiences from episodic memory.

    This finds past tasks similar to the current one to provide
    few-shot guidance to the agent.
    """
    print("[Memory] Searching for similar past experiences...")

    memory = get_episodic_memory()
    current_task = state.get("current_task", "")

    if not current_task:
        print("[Memory] No current task set, skipping episode retrieval")
        return {"similar_episodes": []}

    episodes = memory.recall_similar(
        task=current_task,
        user_id=state["user_id"],
        k=3
    )

    print(f"[Memory] Found {len(episodes)} similar episodes")
    for ep in episodes:
        print(f"  - {ep.get('task', 'Unknown')[:50]}... (similarity: {ep.get('similarity', 0):.2f})")

    return {"similar_episodes": episodes}


async def call_agent(state: AgentState) -> dict[str, Any]:
    """Node: Call Claude with the current context and tools.

    This is the main reasoning node where Claude decides
    what to do next.
    """
    print("[Agent] Calling Claude for reasoning...")

    model = get_model()
    model_with_tools = model.bind_tools(TOOLS)

    # Build system prompt with memory context
    system_prompt = build_system_prompt(state)

    # Prepare messages with system prompt
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    # Call the model
    response = await model_with_tools.ainvoke(messages)

    print(f"[Agent] Response type: {type(response).__name__}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[Agent] Tool calls: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": [response]}


async def execute_tools(state: AgentState) -> dict[str, Any]:
    """Node: Execute tool calls from the agent's response.

    This handles all tool execution and returns results back to the agent.
    Tool results are appended as ToolMessages, which the agent will see on the next iteration.

    The node also tracks actions taken for episodic memory storage.
    """
    print("[Tools] Executing tool calls...")

    # Import tool handlers (lazy import to avoid circular dependencies)
    from mcp_server.tools import (
        handle_calculator,
        handle_read_file,
        handle_write_file,
        handle_list_files,
        handle_get_weather,
    )

    # Get the last message (should be AIMessage with tool_calls)
    last_message = state["messages"][-1]

    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        print("[Tools] No tool calls to execute")
        return {"messages": []}

    tool_messages = []
    # Track actions for episodic memory - accumulate with existing actions from previous tool calls
    actions_taken = list(state.get("task_actions", []))
    memory = get_long_term_memory()

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print(f"[Tools] Executing: {tool_name}({json.dumps(tool_args)})")

        # Route to appropriate handler
        try:
            if tool_name == "calculator":
                result = await handle_calculator(tool_args)
                actions_taken.append(f"calculator({tool_args.get('operation')})")

            elif tool_name == "read_file":
                result = await handle_read_file(tool_args)
                actions_taken.append(f"read_file({tool_args.get('filename')})")

            elif tool_name == "write_file":
                result = await handle_write_file(tool_args)
                actions_taken.append(f"write_file({tool_args.get('filename')})")

            elif tool_name == "list_files":
                result = await handle_list_files(tool_args)
                actions_taken.append("list_files()")

            elif tool_name == "get_weather":
                result = await handle_get_weather(tool_args)
                actions_taken.append(f"get_weather({tool_args.get('city')})")

            elif tool_name == "store_user_preference":
                # Handle preference storage
                key = tool_args.get("key", "")
                value = tool_args.get("value", "")
                memory.store_preference(state["user_id"], key, value)
                result = [{"type": "text", "text": f"Stored preference: {key} = {value}"}]
                actions_taken.append(f"store_preference({key})")

            elif tool_name == "store_user_fact":
                # Handle fact storage
                fact_type = tool_args.get("fact_type", "general")
                content = tool_args.get("content", "")
                memory.store_fact(state["user_id"], fact_type, content, source="user_stated")
                result = [{"type": "text", "text": f"Stored fact: {content}"}]
                actions_taken.append(f"store_fact({fact_type})")

            else:
                result = [{"type": "text", "text": f"Unknown tool: {tool_name}"}]

            # Extract text from result
            result_text = result[0]["text"] if result else "No result"

        except Exception as e:
            result_text = f"Error executing {tool_name}: {str(e)}"
            print(f"[Tools] Error: {result_text}")

        print(f"[Tools] Result: {result_text[:100]}...")

        # Create tool message
        tool_messages.append(
            ToolMessage(content=result_text, tool_call_id=tool_id)
        )

    return {
        "messages": tool_messages,
        "task_actions": actions_taken
    }


async def store_episode(state: AgentState) -> dict[str, Any]:
    """Node: Store the completed task as an episode in episodic memory.

    This runs after a task is completed to save the experience
    for future reference.
    """
    # Only store if flagged
    if not state.get("should_store_episode", False):
        print("[Memory] Skipping episode storage (not flagged)")
        return {}

    print("[Memory] Storing episode to episodic memory...")

    memory = get_episodic_memory()

    task = state.get("current_task", "")
    actions = state.get("task_actions", [])
    outcome = state.get("task_outcome", "Task completed")

    if task and actions:
        episode_id = memory.store_episode(
            user_id=state["user_id"],
            task=task,
            actions=actions,
            outcome=outcome,
            success=True
        )
        print(f"[Memory] Stored episode: {episode_id}")
    else:
        print("[Memory] No task/actions to store")

    return {"should_store_episode": False}


def route_agent(state: AgentState) -> Literal["tools", "store", "end"]:
    """Routing function: Determine next step after agent reasoning.

    This implements the conditional logic that creates the agent's tool-calling loop:
    - If tools are requested, execute them and return to the agent
    - If task is complete with actions, store as episode before ending
    - Otherwise, end the workflow

    Returns:
        - "tools": If the agent wants to use tools (creates loop: agent → tools → agent)
        - "store": If the task is complete and should be stored as an episode
        - "end": If done (no tools, no storage needed)
    """
    last_message = state["messages"][-1]

    # Check if agent wants to use tools (priority check - enables the agent-tool loop)
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("[Router] -> tools")
        return "tools"

    # Check if we should store this as an episode in episodic memory
    # Only store if: (1) actions were taken, and (2) storage is flagged (every 3rd interaction)
    if state.get("task_actions") and state.get("should_store_episode"):
        print("[Router] -> store")
        return "store"

    # No tools needed and no episode to store - workflow complete
    print("[Router] -> end")
    return "end"

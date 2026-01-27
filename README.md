# Agentic AI with LangGraph, MCP Tools & Memory Systems

A demonstration of an Agentic AI application that showcases:

- **LangGraph** - Graph-based framework for stateful agent workflows
- **MCP Server** - Tools exposed via Model Context Protocol
- **Three Memory Types**:
  - **Short-term**: Conversation history within a thread
  - **Long-term**: User preferences/facts persisted across sessions (SQLite)
  - **Episodic**: Past experiences recalled for similar tasks (ChromaDB)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LOAD CONTEXT                                                    │
│  ├─ Long-term: Get user preferences & facts from SQLite         │
│  └─ Episodic: Find similar past tasks from ChromaDB             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT REASONING (Claude Haiku)                                  │
│  ├─ System prompt includes: preferences, facts, past examples   │
│  ├─ Short-term: Full conversation history from checkpointer     │
│  └─ Decides: use tools or provide final answer                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────────┐
│  EXECUTE TOOLS (MCP)    │     │  STORE & UPDATE                  │
│  ├─ Calculator          │     │  ├─ Store episode to ChromaDB    │
│  ├─ File operations     │     │  └─ Update facts in SQLite       │
│  └─ Weather lookup      │     └─────────────────────────────────┘
└───────────┬─────────────┘
            │
            └──────► Back to Agent (loop until done)
```

## Prerequisites

### 1. Python 3.10 or higher

This project requires Python 3.10+. Check your version:

```bash
python3 --version
```

**Install Python 3.10+ if needed:**

**macOS (using Homebrew):**
```bash
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
```

**Windows:**
Download from https://www.python.org/downloads/

### 2. Anthropic API Key

Get an API key from https://console.anthropic.com

## Installation

1. **Clone/navigate to the project:**
   ```bash
   cd /Users/venugopaltati/Projects/AgentWorker
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install langgraph langchain-anthropic langchain-core mcp chromadb python-dotenv
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

## Usage

**Run the interactive CLI:**
```bash
python main.py
```

**Available Commands:**
- `/newthread` - Start a new conversation thread
- `/showmemory` - Show stored memories for current user
- `/clearmemory` - Clear all memories for current user
- `/help` - Show help message
- `/quit` - Exit the application

## Demo Scenarios

### 1. Short-Term Memory (Same Conversation)
```
You: My name is Alex and I prefer temperatures in Celsius
Agent: Nice to meet you, Alex! I've noted your preference for Celsius.

You: What's the weather in Miami?
Agent: The weather in Miami is 28°C (showing Celsius as you prefer)...
```

### 2. Long-Term Memory (Across Sessions)
```
# Session 1
You: I work at Acme Corp
Agent: Got it! I'll remember that you work at Acme Corp.

# Session 2 (after /newthread or restart)
You: Remind me where I work
Agent: You work at Acme Corp!
```

### 3. Episodic Memory (Learning from Past)
```
# After completing several math tasks, the agent learns patterns
You: Help me calculate investment returns
Agent: Based on similar calculations I've done before, I'll use...
```

## Project Structure

```
AgentWorker/
├── main.py                 # Interactive CLI entry point
├── pyproject.toml          # Project configuration
├── .env.example            # Environment template
│
├── agent/                  # LangGraph Agent
│   ├── graph.py            # Main agent graph definition
│   ├── state.py            # Agent state schema
│   ├── nodes.py            # Graph nodes
│   └── memory/
│       ├── short_term.py   # Thread/conversation memory
│       ├── long_term.py    # SQLite for user profiles
│       └── episodic.py     # ChromaDB for experiences
│
├── mcp_server/             # MCP Server (Tools)
│   ├── server.py           # Main MCP server
│   └── tools/
│       ├── calculator.py   # Math operations
│       ├── file_ops.py     # File read/write/list
│       └── weather.py      # Mock weather data
│
├── demo/workspace/         # Demo files for file operations
│   ├── notes.txt
│   └── tasks.json
│
└── storage/                # Persistent storage (auto-created)
    ├── long_term.db        # SQLite database
    └── episodic/           # ChromaDB vector store
```

## Available Tools

| Tool | Description |
|------|-------------|
| `calculator` | Basic math: add, subtract, multiply, divide |
| `read_file` | Read files from demo/workspace |
| `write_file` | Write files to demo/workspace |
| `list_files` | List files in demo/workspace |
| `get_weather` | Get weather for a city (mock data) |
| `store_user_preference` | Save user preference to long-term memory |
| `store_user_fact` | Save fact about user to long-term memory |

## Memory Types Explained

### Short-Term Memory
- **What**: Conversation history within a single thread
- **How**: LangGraph's `MemorySaver` checkpointer
- **Scope**: Thread-specific, cleared on `/newthread`
- **Example**: "Earlier you said..." within same conversation

### Long-Term Memory
- **What**: User preferences and facts
- **How**: SQLite database
- **Scope**: User-specific, persists across all sessions
- **Example**: Name, workplace, temperature unit preference

### Episodic Memory
- **What**: Past task executions and experiences
- **How**: ChromaDB vector store with semantic search
- **Scope**: User-specific, enables learning from past
- **Example**: "Last time you asked about X, I did Y..."

## Troubleshooting

**Python version error:**
```
ERROR: Requires Python >=3.10
```
Install Python 3.10+ and recreate the virtual environment.

**API key error:**
```
Error: ANTHROPIC_API_KEY environment variable not set!
```
Make sure you've created `.env` with your API key.

**Import errors:**
Make sure you're running from the project root directory and the virtual environment is activated.

## License

MIT License - Feel free to use and modify for learning purposes.

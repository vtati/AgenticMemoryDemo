"""MCP Server that exposes tools for the agentic AI application.

This server registers calculator, file operations, and weather tools
and communicates via stdio transport.
"""

import asyncio
import sys
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .tools import (
    calculator_tools,
    handle_calculator,
    file_tools,
    handle_read_file,
    handle_write_file,
    handle_list_files,
    weather_tool,
    handle_get_weather,
)


# Create the MCP server instance
server = Server("agent-worker-tools")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return list of available tools."""
    return [
        Tool(
            name=calculator_tools["name"],
            description=calculator_tools["description"],
            inputSchema=calculator_tools["inputSchema"]
        ),
        Tool(
            name=file_tools["read_file"]["name"],
            description=file_tools["read_file"]["description"],
            inputSchema=file_tools["read_file"]["inputSchema"]
        ),
        Tool(
            name=file_tools["write_file"]["name"],
            description=file_tools["write_file"]["description"],
            inputSchema=file_tools["write_file"]["inputSchema"]
        ),
        Tool(
            name=file_tools["list_files"]["name"],
            description=file_tools["list_files"]["description"],
            inputSchema=file_tools["list_files"]["inputSchema"]
        ),
        Tool(
            name=weather_tool["name"],
            description=weather_tool["description"],
            inputSchema=weather_tool["inputSchema"]
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool and return the result."""
    # Route to appropriate handler
    if name == "calculator":
        result = await handle_calculator(arguments)
    elif name == "read_file":
        result = await handle_read_file(arguments)
    elif name == "write_file":
        result = await handle_write_file(arguments)
    elif name == "list_files":
        result = await handle_list_files(arguments)
    elif name == "get_weather":
        result = await handle_get_weather(arguments)
    else:
        result = [{"type": "text", "text": f"Error: Unknown tool '{name}'"}]

    # Convert to TextContent objects
    return [TextContent(type="text", text=item["text"]) for item in result]


async def main():
    """Run the MCP server."""
    print("Starting MCP Tools Server...", file=sys.stderr)
    print("Available tools: calculator, read_file, write_file, list_files, get_weather", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())

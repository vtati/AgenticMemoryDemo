"""File operations tools for reading, writing, and listing files."""

import os
from pathlib import Path
from typing import Any

# Default workspace directory (can be overridden via environment variable)
WORKSPACE_PATH = os.environ.get("WORKSPACE_PATH", "demo/workspace")


def _get_workspace_path() -> Path:
    """Get the absolute path to the workspace directory."""
    # Resolve relative to the project root
    base_path = Path(__file__).parent.parent.parent
    return (base_path / WORKSPACE_PATH).resolve()


def _safe_path(filename: str) -> Path:
    """Ensure the path is within the workspace directory (security).

    Args:
        filename: Relative filename within workspace

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If path attempts to escape workspace
    """
    workspace = _get_workspace_path()
    resolved = (workspace / filename).resolve()

    # Security check: ensure path is within workspace
    if not str(resolved).startswith(str(workspace)):
        raise ValueError("Access denied: Path is outside workspace directory")

    return resolved


# Tool definitions for MCP registration
file_tools = {
    "read_file": {
        "name": "read_file",
        "description": "Read the contents of a file from the workspace directory",
        "inputSchema": {
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
    "write_file": {
        "name": "write_file",
        "description": "Write content to a file in the workspace directory",
        "inputSchema": {
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
    "list_files": {
        "name": "list_files",
        "description": "List all files in the workspace directory",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}


async def handle_read_file(arguments: dict[str, Any]) -> list[dict]:
    """Read a file from the workspace.

    Args:
        arguments: Dict with 'filename' key

    Returns:
        List with text content block containing file contents or error
    """
    filename = arguments.get("filename")

    if not filename:
        return [{"type": "text", "text": "Error: Missing required argument 'filename'"}]

    try:
        file_path = _safe_path(filename)

        if not file_path.exists():
            return [{"type": "text", "text": f"Error: File '{filename}' not found"}]

        if not file_path.is_file():
            return [{"type": "text", "text": f"Error: '{filename}' is not a file"}]

        content = file_path.read_text(encoding="utf-8")
        return [{"type": "text", "text": content}]

    except ValueError as e:
        return [{"type": "text", "text": str(e)}]
    except Exception as e:
        return [{"type": "text", "text": f"Error reading file: {e}"}]


async def handle_write_file(arguments: dict[str, Any]) -> list[dict]:
    """Write content to a file in the workspace.

    Args:
        arguments: Dict with 'filename' and 'content' keys

    Returns:
        List with text content block confirming write or error
    """
    filename = arguments.get("filename")
    content = arguments.get("content")

    if not filename:
        return [{"type": "text", "text": "Error: Missing required argument 'filename'"}]

    if content is None:
        return [{"type": "text", "text": "Error: Missing required argument 'content'"}]

    try:
        file_path = _safe_path(filename)

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(content, encoding="utf-8")
        return [{"type": "text", "text": f"Successfully wrote to '{filename}'"}]

    except ValueError as e:
        return [{"type": "text", "text": str(e)}]
    except Exception as e:
        return [{"type": "text", "text": f"Error writing file: {e}"}]


async def handle_list_files(arguments: dict[str, Any]) -> list[dict]:
    """List all files in the workspace directory.

    Args:
        arguments: Empty dict (no arguments needed)

    Returns:
        List with text content block containing file listing
    """
    try:
        workspace = _get_workspace_path()

        if not workspace.exists():
            return [{"type": "text", "text": f"Workspace directory does not exist: {workspace}"}]

        files = []
        for item in sorted(workspace.iterdir()):
            if item.is_file():
                files.append(f"  {item.name}")
            elif item.is_dir():
                files.append(f"  {item.name}/")

        if not files:
            return [{"type": "text", "text": "Workspace is empty"}]

        file_list = "\n".join(files)
        return [{"type": "text", "text": f"Files in workspace:\n{file_list}"}]

    except Exception as e:
        return [{"type": "text", "text": f"Error listing files: {e}"}]

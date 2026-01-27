"""Calculator tool for basic arithmetic operations."""

from typing import Any

# Tool definition for MCP registration
calculator_tools = {
    "name": "calculator",
    "description": "Perform basic arithmetic operations (add, subtract, multiply, divide) on two numbers",
    "inputSchema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform"
            },
            "a": {
                "type": "number",
                "description": "The first operand"
            },
            "b": {
                "type": "number",
                "description": "The second operand"
            }
        },
        "required": ["operation", "a", "b"]
    }
}


async def handle_calculator(arguments: dict[str, Any]) -> list[dict]:
    """Execute calculator operation and return result.

    Args:
        arguments: Dict with 'operation', 'a', and 'b' keys

    Returns:
        List with single text content block containing the result
    """
    operation = arguments.get("operation")
    a = arguments.get("a")
    b = arguments.get("b")

    # Validate inputs
    if operation is None or a is None or b is None:
        return [{"type": "text", "text": "Error: Missing required arguments (operation, a, b)"}]

    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return [{"type": "text", "text": "Error: Arguments 'a' and 'b' must be numbers"}]

    # Perform calculation
    result: float
    operation_symbol: str

    if operation == "add":
        result = a + b
        operation_symbol = "+"
    elif operation == "subtract":
        result = a - b
        operation_symbol = "-"
    elif operation == "multiply":
        result = a * b
        operation_symbol = "*"
    elif operation == "divide":
        if b == 0:
            return [{"type": "text", "text": "Error: Division by zero is not allowed"}]
        result = a / b
        operation_symbol = "/"
    else:
        return [{"type": "text", "text": f"Error: Unknown operation '{operation}'"}]

    # Format result (remove trailing zeros for clean output)
    if result == int(result):
        result_str = str(int(result))
    else:
        result_str = f"{result:.6f}".rstrip('0').rstrip('.')

    return [{"type": "text", "text": f"{a} {operation_symbol} {b} = {result_str}"}]

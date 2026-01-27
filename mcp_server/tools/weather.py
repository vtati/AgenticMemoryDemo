"""Weather tool providing mock weather data for demonstration."""

import random
from typing import Any

# Mock weather data for demo cities
MOCK_WEATHER_DATA: dict[str, dict] = {
    "new york": {"temp_f": 45, "temp_c": 7, "condition": "Cloudy", "humidity": 65},
    "los angeles": {"temp_f": 72, "temp_c": 22, "condition": "Sunny", "humidity": 40},
    "chicago": {"temp_f": 38, "temp_c": 3, "condition": "Windy", "humidity": 55},
    "miami": {"temp_f": 82, "temp_c": 28, "condition": "Partly Cloudy", "humidity": 75},
    "seattle": {"temp_f": 52, "temp_c": 11, "condition": "Rainy", "humidity": 80},
    "denver": {"temp_f": 55, "temp_c": 13, "condition": "Clear", "humidity": 30},
    "san francisco": {"temp_f": 58, "temp_c": 14, "condition": "Foggy", "humidity": 70},
    "boston": {"temp_f": 42, "temp_c": 6, "condition": "Overcast", "humidity": 60},
    "austin": {"temp_f": 78, "temp_c": 26, "condition": "Sunny", "humidity": 45},
    "portland": {"temp_f": 50, "temp_c": 10, "condition": "Drizzle", "humidity": 75},
}

# Tool definition for MCP registration
weather_tool = {
    "name": "get_weather",
    "description": "Get the current weather for a city. Returns temperature (Fahrenheit and Celsius), condition, and humidity.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city to get weather for"
            }
        },
        "required": ["city"]
    }
}


async def handle_get_weather(arguments: dict[str, Any]) -> list[dict]:
    """Get weather data for a city.

    Args:
        arguments: Dict with 'city' key

    Returns:
        List with text content block containing weather data as JSON
    """
    city = arguments.get("city")

    if not city:
        return [{"type": "text", "text": "Error: Missing required argument 'city'"}]

    city_lower = city.lower().strip()

    # Check if we have data for this city
    if city_lower in MOCK_WEATHER_DATA:
        weather = MOCK_WEATHER_DATA[city_lower]
        result = {
            "city": city,
            "temperature_fahrenheit": f"{weather['temp_f']}°F",
            "temperature_celsius": f"{weather['temp_c']}°C",
            "condition": weather["condition"],
            "humidity": f"{weather['humidity']}%"
        }
    else:
        # Generate random weather for unknown cities
        temp_f = random.randint(30, 90)
        temp_c = round((temp_f - 32) * 5 / 9)
        conditions = ["Sunny", "Cloudy", "Rainy", "Clear", "Windy", "Partly Cloudy"]
        condition = random.choice(conditions)
        humidity = random.randint(30, 80)

        result = {
            "city": city,
            "temperature_fahrenheit": f"{temp_f}°F",
            "temperature_celsius": f"{temp_c}°C",
            "condition": condition,
            "humidity": f"{humidity}%",
            "note": "Weather data simulated (city not in database)"
        }

    # Format as readable text
    output_lines = [
        f"Weather for {result['city']}:",
        f"  Temperature: {result['temperature_fahrenheit']} ({result['temperature_celsius']})",
        f"  Condition: {result['condition']}",
        f"  Humidity: {result['humidity']}"
    ]

    if "note" in result:
        output_lines.append(f"  Note: {result['note']}")

    return [{"type": "text", "text": "\n".join(output_lines)}]

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .toolkit import Toolkit


def sleep_tool(seconds: float) -> str:
    time.sleep(seconds)
    return f"slept {seconds}s"

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "execute_calculator",
        "description": "Perform numerical calculations.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A string that represents the calculation to be performed.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "type": "function",
        "name": "execute_code",
        "description": "Execute a block of code.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A block of code that is to be executed.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "type": "function",
        "name": "execute_pdf_extractions",
        "description": "Extract key details and requested information from a pdf.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A string of information to extract a pdf.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "type": "function",
        "name": "sleep_tool",
        "description": "Sleep for N seconds and return a message. Useful for testing parallelism.",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number"}
            },
            "required": ["seconds"],
        },
    },
]

TOOL_MAP: dict[str, Callable[..., Any]] = {
    "execute_calculator": Toolkit.execute_calculator,
    "execute_code": Toolkit.execute_code,
    "execute_pdf_extractions": Toolkit.execute_pdf_extractions,
    "sleep_tool": sleep_tool,
}


def run_tool(name: str, arguments: Any) -> Any:
    function = TOOL_MAP.get(name)
    if not function:
        raise ValueError(f"Tool not found: {name}")

    if arguments is None:
        arguments = {}

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if not isinstance(arguments, dict):
        raise TypeError(f"Tool arguments must be dict, got: {type(arguments)} ({arguments})")

    return function(**arguments)

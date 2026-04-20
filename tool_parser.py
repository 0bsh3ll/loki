"""
LOKI Tool Parser

Parses the LLM's text response to detect tool invocations.
The LLM is instructed to use the format:

    tool: TOOL_NAME({"arg1": "value1", "arg2": "value2"})

This module extracts those lines and returns structured data
the agent loop can act on. Everything else is treated as
regular assistant text.
"""

import json,re
from typing import Any, Dict, List, Tuple

from config import TOOL_CALL_PREFIX

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_tool_invocations(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Scans the LLM response text for tool call lines.

    Returns a list of (tool_name, args_dict) tuples.
    Malformed lines are silently skipped — no crashes.

    Example:
        Input:  "Let me read that.\ntool: read_file({\"filename\": \"x.py\"})"
        Output: [("read_file", {"filename": "x.py"})]
    """
    invocations = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # Only process lines that start with the tool prefix
        if not line.lower().startswith(TOOL_CALL_PREFIX):
            continue

        try:
            # Strip the "tool:" prefix
            after_prefix = line[len(TOOL_CALL_PREFIX):].strip()

            # Split on the first "(" to get tool name and args
            paren_idx = after_prefix.index("(")
            tool_name = after_prefix[:paren_idx].strip()

            # Everything after the "(" up to the last ")" is JSON
            rest = after_prefix[paren_idx + 1:]
            if not rest.endswith(")"):
                continue

            json_str = rest[:-1].strip()

            # Parse the JSON arguments
            args = json.loads(json_str)

            if not isinstance(args, dict):
                continue

            invocations.append((tool_name, args))

        except (ValueError, json.JSONDecodeError, IndexError):
            # Malformed line — skip it, don't crash
            continue

    return invocations


def has_tool_calls(text: str) -> bool:
    """Quick check: does this response contain any tool calls?"""
    return len(extract_tool_invocations(text)) > 0


def strip_tool_lines(text: str) -> str:
    """
    Returns the response text with all tool call lines removed.
    Useful for extracting the 'prose' part of a mixed response
    where the LLM writes some text AND requests a tool.
    """
    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip().lower().startswith(TOOL_CALL_PREFIX):
            lines.append(raw_line)

    # Clean up leading/trailing blank lines from removal
    result = "\n".join(lines).strip()
    return result

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


# Lines that are pure code-fence markers (```python, ```, etc.) — strip
# these so a model that wraps its tool call in a fence still gets parsed.
_CODE_FENCE_RE = re.compile(r"^\s*```\w*\s*$")


def _clean(text: str) -> str:
    """Drop code-fence marker lines so wrapped tool calls still parse."""
    return "\n".join(
        line for line in text.splitlines() if not _CODE_FENCE_RE.match(line)
    )


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _try_parse_call(after_prefix: str) -> Tuple[str, Dict[str, Any]] | None:
    """Parse 'name({...})' → (name, args). Returns None on malformed."""
    try:
        paren_idx = after_prefix.index("(")
    except ValueError:
        return None
    tool_name = after_prefix[:paren_idx].strip()
    if not tool_name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tool_name):
        return None
    rest = after_prefix[paren_idx + 1:].rstrip()
    if not rest.endswith(")"):
        return None
    json_str = rest[:-1].strip()
    try:
        args = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    return tool_name, args


def extract_tool_invocations(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Scans the LLM response text for tool call lines.

    Accepts (forgiving on purpose, since 7B-class models drift):
      - ``tool: NAME({...})``                        ← canonical
      - ``NAME({...})`` on its own line              ← prefix dropped
      - lines wrapped in ```...``` code fences       ← fence stripped
      - leading bullet/quote noise like "- " or "> "

    Returns a list of (tool_name, args_dict) tuples.
    Malformed lines are silently skipped — no crashes.
    """
    invocations: List[Tuple[str, Dict[str, Any]]] = []

    for raw_line in _clean(text).splitlines():
        line = raw_line.strip()
        # Strip leading list/quote/bold markers a model might add.
        line = re.sub(r"^[-*>\s`]+", "", line)
        line = line.strip("`*_ ")

        if not line:
            continue

        # Canonical: starts with "tool:".
        if line.lower().startswith(TOOL_CALL_PREFIX):
            after_prefix = line[len(TOOL_CALL_PREFIX):].strip()
            parsed = _try_parse_call(after_prefix)
            if parsed:
                invocations.append(parsed)
            continue

        # Fallback: bare "NAME({...})" on its own line.
        # Only accept when the whole line is a single call — avoids
        # picking up python snippets the model wrote in prose.
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(.*\)\s*$", line):
            parsed = _try_parse_call(line)
            if parsed:
                invocations.append(parsed)

    return invocations


def has_tool_calls(text: str) -> bool:
    """Quick check: does this response contain any tool calls?"""
    return len(extract_tool_invocations(text)) > 0


def strip_tool_lines(text: str) -> str:
    """
    Returns the response text with all tool call lines removed.
    Useful for extracting the 'prose' part of a mixed response
    where the LLM writes some text AND requests a tool.
    Mirrors extract_tool_invocations' tolerance for code fences and
    bare ``name({...})`` lines, so the prose printed back to the user
    doesn't include the call.
    """
    lines = []
    for raw_line in _clean(text).splitlines():
        stripped = re.sub(r"^[-*>\s`]+", "", raw_line.strip()).strip("`*_ ")

        if stripped.lower().startswith(TOOL_CALL_PREFIX):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(.*\)\s*$", stripped):
            continue
        lines.append(raw_line)

    return "\n".join(lines).strip()

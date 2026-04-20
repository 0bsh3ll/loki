"""
LOKI Tool Registry

The central hub for all tools. Tools register themselves using the
@register_tool decorator. The agent loop never touches individual
tool files — it only talks to this registry.

Usage:
    from tool_registry import register_tool, get_tool, execute_tool, get_all_tool_schemas

    @register_tool("my_tool")
    def my_tool(arg1: str, arg2: int = 5) -> dict:
        '''Does something useful.
        :param arg1: First argument.
        :param arg2: Second argument.
        '''
        return {"result": "done"}
"""

import inspect
from typing import Any, Callable, Dict, List, Optional


class ToolEntry:
    """Holds everything we know about a registered tool."""

    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description
        self.signature = inspect.signature(func)

    def get_schema_string(self) -> str:
        """
        Returns a human-readable schema block for injection into
        the system prompt. Auto-generated from the function's
        signature and docstring.
        """
        params = []
        for param_name, param in self.signature.parameters.items():
            type_hint = (
                param.annotation.__name__
                if param.annotation != inspect.Parameter.empty
                else "any"
            )
            default = (
                f" (default: {param.default!r})"
                if param.default != inspect.Parameter.empty
                else " (required)"
            )
            params.append(f"    - {param_name}: {type_hint}{default}")

        params_block = "\n".join(params) if params else "    (none)"

        return (
            f"TOOL: {self.name}\n"
            f"Description: {self.description}\n"
            f"Parameters:\n{params_block}\n"
            f"Signature: {self.name}{self.signature}"
        )


# ─── The Registry ──────────────────────────────────────────────────

_REGISTRY: Dict[str, ToolEntry] = {}


def register_tool(name: str, description: Optional[str] = None):
    """
    Decorator that registers a function as a tool.

    If no description is provided, the function's docstring is used.
    The first line of the docstring becomes the description.

    @register_tool("read_file")
    def read_file_tool(filename: str) -> dict:
        '''Reads a file and returns its contents.
        :param filename: Path to the file.
        '''
        ...
    """

    def decorator(func: Callable) -> Callable:
        desc = description
        if desc is None:
            # Extract first line of docstring as description
            doc = inspect.getdoc(func)
            desc = doc.split("\n")[0].strip() if doc else f"Tool: {name}"

        entry = ToolEntry(name=name, func=func, description=desc)
        _REGISTRY[name] = entry
        return func

    return decorator


def get_tool(name: str) -> Optional[ToolEntry]:
    """Returns a ToolEntry by name, or None if not found."""
    return _REGISTRY.get(name)


def get_all_tools() -> Dict[str, ToolEntry]:
    """Returns the full registry dict."""
    return dict(_REGISTRY)


def get_tool_names() -> List[str]:
    """Returns a list of all registered tool names."""
    return list(_REGISTRY.keys())


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a registered tool by name with the given arguments.

    Returns the tool's result dict on success.
    Returns an error dict if the tool is not found or execution fails.
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        return {"error": f"Unknown tool: {name}"}

    try:
        result = entry.func(**args)
        return result
    except Exception as e:
        return {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}


def get_all_tool_schemas() -> str:
    """
    Returns a formatted string block describing every registered tool.
    This gets injected into the system prompt so the LLM knows what
    tools are available and how to call them.
    """
    if not _REGISTRY:
        return "(No tools registered)"

    blocks = []
    for entry in _REGISTRY.values():
        blocks.append(entry.get_schema_string())

    separator = "\n" + "=" * 40 + "\n"
    return separator.join(blocks)


def clear_registry():
    """Clears all registered tools. Used in tests."""
    _REGISTRY.clear()

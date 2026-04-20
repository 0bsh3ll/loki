"""
Tool: read_file

Reads the full content of a file and returns it as structured data.
"""

from pathlib import Path
from typing import Any, Dict

from tool_registry import register_tool


def _resolve_path(path_str: str) -> Path:
    """Resolve a path string to an absolute Path."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


@register_tool("read_file")
def read_file(filename: str) -> Dict[str, Any]:
    """Reads the full content of a file and returns it.
    :param filename: The path to the file to read.
    :return: A dict with file_path and content, or an error.
    """
    full_path = _resolve_path(filename)

    if not full_path.exists():
        return {
            "error": f"File not found: {full_path}"
        }

    if not full_path.is_file():
        return {
            "error": f"Not a file: {full_path}"
        }

    try:
        content = full_path.read_text(encoding="utf-8")
        return {
            "file_path": str(full_path),
            "content": content
        }
    except Exception as e:
        return {
            "error": f"Failed to read {full_path}: {type(e).__name__}: {e}"
        }

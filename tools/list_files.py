"""
Tool: list_files

Lists the files and directories inside a given path.
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


@register_tool("list_files")
def list_files(path: str = ".") -> Dict[str, Any]:
    """Lists the files and subdirectories in a directory.
    :param path: The directory path to list. Defaults to current directory.
    :return: A dict with the path and a list of file entries.
    """
    full_path = _resolve_path(path)

    if not full_path.exists():
        return {
            "error": f"Path not found: {full_path}"
        }

    if not full_path.is_dir():
        return {
            "error": f"Not a directory: {full_path}"
        }

    try:
        entries = []
        for item in sorted(full_path.iterdir()):
            # Skip hidden files/dirs
            if item.name.startswith("."):
                continue
            entries.append({
                "name": item.name,
                "type": "file" if item.is_file() else "dir"
            })

        return {
            "path": str(full_path),
            "files": entries
        }
    except Exception as e:
        return {
            "error": f"Failed to list {full_path}: {type(e).__name__}: {e}"
        }

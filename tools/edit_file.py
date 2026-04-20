"""
Tool: edit_file

Creates a new file or edits an existing one via find-and-replace.
- If old_str is empty: creates/overwrites the file with new_str.
- If old_str is provided: finds the first occurrence and replaces it.
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


@register_tool("edit_file")
def edit_file(path: str, new_str: str, old_str: str = "") -> Dict[str, Any]:
    """Creates a new file or replaces text in an existing file.
    If old_str is empty, the file is created (or overwritten) with new_str.
    If old_str is provided, the first occurrence in the file is replaced with new_str.
    :param path: The path to the file to create or edit.
    :param new_str: The new content to write or the replacement string.
    :param old_str: The string to find and replace. Empty means create/overwrite.
    :return: A dict with the path and the action taken.
    """
    full_path = _resolve_path(path)

    try:
        # ── Create / Overwrite mode ────────────────────────────
        if old_str == "":
            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(new_str, encoding="utf-8")
            return {
                "path": str(full_path),
                "action": "created_file"
            }

        # ── Find and Replace mode ──────────────────────────────
        if not full_path.exists():
            return {
                "error": f"File not found: {full_path}"
            }

        original = full_path.read_text(encoding="utf-8")

        if old_str not in original:
            return {
                "path": str(full_path),
                "action": "old_str_not_found"
            }

        # Replace only the first occurrence
        edited = original.replace(old_str, new_str, 1)
        full_path.write_text(edited, encoding="utf-8")

        return {
            "path": str(full_path),
            "action": "edited"
        }

    except Exception as e:
        return {
            "error": f"Failed to edit {full_path}: {type(e).__name__}: {e}"
        }

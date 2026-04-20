"""
LOKI Test Suite

Run: python -m tests.test_tools  (from loki/ directory)
  or: python tests/test_tools.py

Tests every component in isolation — no LLM needed.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

# ─── Make sure imports resolve from project root ───────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOOL_CALL_PREFIX, COLOR_USER, COLOR_RESET, MODEL_NAME
from tool_registry import (
    register_tool,
    get_tool,
    get_all_tools,
    get_tool_names,
    execute_tool,
    get_all_tool_schemas,
    clear_registry,
)
from tool_parser import extract_tool_invocations, has_tool_calls, strip_tool_lines


# ═══════════════════════════════════════════════════════════════════
# CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):
    """Sanity checks on config values."""

    def test_model_name_is_set(self):
        self.assertIsInstance(MODEL_NAME, str)
        self.assertGreater(len(MODEL_NAME), 0)

    def test_tool_call_prefix(self):
        self.assertEqual(TOOL_CALL_PREFIX, "tool:")

    def test_colors_are_strings(self):
        self.assertIsInstance(COLOR_USER, str)
        self.assertIsInstance(COLOR_RESET, str)

    def test_allow_thinking_exists(self):
        """ALLOW_THINKING flag should exist in config."""
        from config import ALLOW_THINKING
        self.assertIsInstance(ALLOW_THINKING, bool)


# ═══════════════════════════════════════════════════════════════════
# TOOL REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════

class TestToolRegistry(unittest.TestCase):
    """Tests for the decorator-based tool registry."""

    def setUp(self):
        clear_registry()

    def tearDown(self):
        clear_registry()

    # ─── Registration ──────────────────────────────────────────────

    def test_decorator_registers_tool(self):
        """A decorated function should appear in the registry."""
        @register_tool("greet")
        def greet_tool(name: str) -> dict:
            """Says hello."""
            return {"message": f"Hello, {name}"}
        self.assertIn("greet", get_tool_names())
        self.assertIsNotNone(get_tool("greet"))

    def test_decorator_uses_docstring_as_description(self):
        """If no explicit description, first docstring line is used."""
        @register_tool("demo")
        def demo_tool() -> dict:
            """This is the description line.
            More details here.
            """
            return {}
        entry = get_tool("demo")
        self.assertEqual(entry.description, "This is the description line.")

    def test_decorator_with_explicit_description(self):
        """Explicit description overrides docstring."""
        @register_tool("demo2", description="Custom description")
        def demo2_tool() -> dict:
            """Docstring description."""
            return {}
        entry = get_tool("demo2")
        self.assertEqual(entry.description, "Custom description")

    def test_multiple_tools_registered(self):
        """Multiple tools can coexist in the registry."""
        @register_tool("tool_a")
        def a() -> dict:
            """Tool A."""
            return {}
        @register_tool("tool_b")
        def b() -> dict:
            """Tool B."""
            return {}
        @register_tool("tool_c")
        def c() -> dict:
            """Tool C."""
            return {}
        self.assertEqual(len(get_all_tools()), 3)
        self.assertEqual(sorted(get_tool_names()), ["tool_a", "tool_b", "tool_c"])

    def test_get_tool_not_found(self):
        """Looking up a non-existent tool returns None."""
        self.assertIsNone(get_tool("does_not_exist"))

    def test_clear_registry(self):
        """clear_registry() empties everything."""
        @register_tool("temp")
        def temp() -> dict:
            """Temporary."""
            return {}
        self.assertEqual(len(get_all_tools()), 1)
        clear_registry()
        self.assertEqual(len(get_all_tools()), 0)

    # ─── Schema Generation ─────────────────────────────────────────

    def test_schema_contains_tool_name(self):
        @register_tool("finder")
        def finder(path: str) -> dict:
            """Finds things."""
            return {}
        schema = get_all_tool_schemas()
        self.assertIn("finder", schema)

    def test_schema_contains_description(self):
        @register_tool("searcher")
        def searcher(query: str) -> dict:
            """Searches the web for information."""
            return {}
        schema = get_all_tool_schemas()
        self.assertIn("Searches the web for information", schema)

    def test_schema_contains_parameter_info(self):
        @register_tool("editor")
        def editor(path: str, content: str, force: bool = False) -> dict:
            """Edits a file."""
            return {}
        schema = get_all_tool_schemas()
        self.assertIn("path", schema)
        self.assertIn("str", schema)
        self.assertIn("force", schema)
        self.assertIn("(default: False)", schema)
        self.assertIn("(required)", schema)

    def test_schema_empty_registry(self):
        schema = get_all_tool_schemas()
        self.assertIn("No tools registered", schema)

    def test_schema_multiple_tools(self):
        @register_tool("alpha")
        def alpha() -> dict:
            """Alpha tool."""
            return {}
        @register_tool("beta")
        def beta() -> dict:
            """Beta tool."""
            return {}
        schema = get_all_tool_schemas()
        self.assertIn("alpha", schema)
        self.assertIn("beta", schema)
        self.assertIn("=" * 40, schema)

    # ─── Execution ─────────────────────────────────────────────────

    def test_execute_tool_success(self):
        @register_tool("adder")
        def adder(a: int, b: int) -> dict:
            """Adds two numbers."""
            return {"sum": a + b}
        result = execute_tool("adder", {"a": 3, "b": 7})
        self.assertEqual(result, {"sum": 10})

    def test_execute_tool_unknown(self):
        result = execute_tool("nonexistent", {})
        self.assertIn("error", result)
        self.assertIn("Unknown tool", result["error"])

    def test_execute_tool_bad_args(self):
        @register_tool("strict")
        def strict(required_arg: str) -> dict:
            """Needs an argument."""
            return {"got": required_arg}
        result = execute_tool("strict", {"wrong_arg": "oops"})
        self.assertIn("error", result)
        self.assertIn("failed", result["error"])

    def test_execute_tool_exception_handling(self):
        @register_tool("crasher")
        def crasher() -> dict:
            """Will crash."""
            raise RuntimeError("Boom!")
        result = execute_tool("crasher", {})
        self.assertIn("error", result)
        self.assertIn("RuntimeError", result["error"])
        self.assertIn("Boom!", result["error"])

    def test_execute_tool_with_defaults(self):
        @register_tool("greeter")
        def greeter(name: str, shout: bool = False) -> dict:
            """Greets someone."""
            msg = f"Hello, {name}!"
            if shout:
                msg = msg.upper()
            return {"message": msg}
        result = execute_tool("greeter", {"name": "Loki"})
        self.assertEqual(result, {"message": "Hello, Loki!"})
        result = execute_tool("greeter", {"name": "Loki", "shout": True})
        self.assertEqual(result, {"message": "HELLO, LOKI!"})


# ═══════════════════════════════════════════════════════════════════
# TOOL PARSER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestToolParser(unittest.TestCase):
    """Tests for parsing tool calls from LLM text output."""

    # ─── Single Call ───────────────────────────────────────────────

    def test_single_tool_call(self):
        text = 'tool: read_file({"filename": "main.py"})'
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "read_file")
        self.assertEqual(result[0][1], {"filename": "main.py"})

    def test_single_call_with_multiple_args(self):
        text = 'tool: edit_file({"path": "x.py", "old_str": "foo", "new_str": "bar"})'
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)
        name, args = result[0]
        self.assertEqual(name, "edit_file")
        self.assertEqual(args["path"], "x.py")
        self.assertEqual(args["old_str"], "foo")
        self.assertEqual(args["new_str"], "bar")

    def test_single_call_with_integer_arg(self):
        text = 'tool: web_search({"query": "python", "num_results": 5})'
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1]["num_results"], 5)

    # ─── Multiple Calls ───────────────────────────────────────────

    def test_multiple_tool_calls(self):
        text = (
            'tool: read_file({"filename": "a.py"})\n'
            'tool: read_file({"filename": "b.py"})'
        )
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1]["filename"], "a.py")
        self.assertEqual(result[1][1]["filename"], "b.py")

    # ─── No Tool Calls ────────────────────────────────────────────

    def test_no_tool_calls_plain_text(self):
        text = "Sure, I can help you with that. Let me explain..."
        result = extract_tool_invocations(text)
        self.assertEqual(result, [])

    def test_no_tool_calls_empty_string(self):
        result = extract_tool_invocations("")
        self.assertEqual(result, [])

    # ─── Mixed Content ─────────────────────────────────────────────

    def test_mixed_prose_and_tool_call(self):
        text = (
            "Let me read that file for you.\n"
            'tool: read_file({"filename": "config.py"})\n'
            "I'll review it now."
        )
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "read_file")

    # ─── Malformed Inputs ──────────────────────────────────────────

    def test_malformed_json(self):
        text = "tool: read_file({broken json here})"
        result = extract_tool_invocations(text)
        self.assertEqual(result, [])

    def test_missing_closing_paren(self):
        text = 'tool: read_file({"filename": "x.py"}'
        result = extract_tool_invocations(text)
        self.assertEqual(result, [])

    def test_no_parentheses(self):
        text = "tool: read_file"
        result = extract_tool_invocations(text)
        self.assertEqual(result, [])

    def test_non_dict_json(self):
        text = 'tool: read_file(["not", "a", "dict"])'
        result = extract_tool_invocations(text)
        self.assertEqual(result, [])

    def test_empty_args(self):
        text = "tool: list_files({})"
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("list_files", {}))

    # ─── Whitespace Handling ───────────────────────────────────────

    def test_leading_whitespace(self):
        text = '   tool: read_file({"filename": "x.py"})'
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)

    def test_spaces_around_tool_name(self):
        text = 'tool:   read_file  ({"filename": "x.py"})'
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "read_file")

    # ─── has_tool_calls ────────────────────────────────────────────

    def test_has_tool_calls_true(self):
        text = 'tool: read_file({"filename": "x.py"})'
        self.assertTrue(has_tool_calls(text))

    def test_has_tool_calls_false(self):
        text = "Just a normal response."
        self.assertFalse(has_tool_calls(text))

    # ─── strip_tool_lines ──────────────────────────────────────────

    def test_strip_tool_lines_removes_calls(self):
        text = (
            "Let me check that.\n"
            'tool: read_file({"filename": "x.py"})\n'
            "Done!"
        )
        stripped = strip_tool_lines(text)
        self.assertNotIn("tool:", stripped)
        self.assertIn("Let me check that.", stripped)
        self.assertIn("Done!", stripped)

    def test_strip_tool_lines_all_tools(self):
        text = (
            'tool: read_file({"filename": "a.py"})\n'
            'tool: read_file({"filename": "b.py"})'
        )
        stripped = strip_tool_lines(text)
        self.assertEqual(stripped, "")

    def test_strip_tool_lines_no_tools(self):
        text = "Hello, I'm Loki."
        stripped = strip_tool_lines(text)
        self.assertEqual(stripped, text)

    # ─── strip_think_tags ──────────────────────────────────────────

    def test_strip_think_tags_removes_block(self):
        """<think>...</think> should be stripped, leaving only the answer."""
        from tool_parser import strip_think_tags
        text = "<think>Let me reason about this...</think>Hello!"
        result = strip_think_tags(text)
        self.assertEqual(result, "Hello!")

    def test_strip_think_tags_no_tags(self):
        """Plain text without think tags should pass through unchanged."""
        from tool_parser import strip_think_tags
        text = "Just a normal response."
        result = strip_think_tags(text)
        self.assertEqual(result, "Just a normal response.")

    def test_strip_think_tags_multiline(self):
        """Multi-line think blocks should be fully stripped."""
        from tool_parser import strip_think_tags
        text = (
            "<think>\n"
            "The user wants to know about Python.\n"
            "I should explain clearly.\n"
            "</think>\n"
            "Python is a programming language."
        )
        result = strip_think_tags(text)
        self.assertEqual(result, "Python is a programming language.")

    def test_strip_think_tags_with_tool_call(self):
        """Think block before a tool call should be stripped, tool call preserved."""
        from tool_parser import strip_think_tags
        text = (
            "<think>I need to read this file first.</think>\n"
            'tool: read_file({"filename": "main.py"})'
        )
        result = strip_think_tags(text)
        self.assertNotIn("<think>", result)
        self.assertIn("tool: read_file", result)

        # Parser should still find the tool call
        invocations = extract_tool_invocations(result)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(invocations[0][0], "read_file")

    def test_strip_think_tags_empty_result(self):
        """Response that is ONLY a think block should return empty string."""
        from tool_parser import strip_think_tags
        text = "<think>Just thinking, no answer.</think>"
        result = strip_think_tags(text)
        self.assertEqual(result, "")

    def test_strip_think_tags_multiple_blocks(self):
        """Multiple think blocks in one response should all be stripped."""
        from tool_parser import strip_think_tags
        text = "<think>first thought</think>Hello <think>second thought</think>World"
        result = strip_think_tags(text)
        self.assertEqual(result, "Hello World")

    def test_strip_think_tags_nested_content(self):
        """Think blocks with special characters inside should still be stripped."""
        from tool_parser import strip_think_tags
        text = '<think>What about {"key": "value"}?</think>The answer is 42.'
        result = strip_think_tags(text)
        self.assertEqual(result, "The answer is 42.")


# ═══════════════════════════════════════════════════════════════════
# AUTO-DISCOVERY TESTS
# ═══════════════════════════════════════════════════════════════════

class TestAutoDiscovery(unittest.TestCase):

    def test_tools_auto_registered(self):
        """Importing the tools package should register all 4 tools."""
        import tools  # noqa: F401
        from tool_registry import get_tool_names
        names = get_tool_names()
        self.assertIn("read_file", names)
        self.assertIn("list_files", names)
        self.assertIn("edit_file", names)
        self.assertIn("web_search", names)

    def test_tools_are_callable(self):
        import tools  # noqa: F401
        from tool_registry import get_tool
        for name in ["read_file", "list_files", "edit_file", "web_search"]:
            entry = get_tool(name)
            self.assertIsNotNone(entry, f"{name} not found in registry")
            self.assertTrue(callable(entry.func), f"{name} is not callable")


# ═══════════════════════════════════════════════════════════════════
# READ_FILE TOOL TESTS
# ═══════════════════════════════════════════════════════════════════

class TestReadFile(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "hello.txt")
        with open(self.test_file, "w") as f:
            f.write("Hello, Loki!")
        self.multiline_file = os.path.join(self.test_dir, "multi.txt")
        with open(self.multiline_file, "w") as f:
            f.write("line 1\nline 2\nline 3\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _get_tool(self):
        import tools  # noqa: F401
        from tool_registry import get_tool
        return get_tool("read_file").func

    def test_read_existing_file(self):
        read = self._get_tool()
        result = read(self.test_file)
        self.assertNotIn("error", result)
        self.assertEqual(result["content"], "Hello, Loki!")

    def test_read_multiline_file(self):
        read = self._get_tool()
        result = read(self.multiline_file)
        self.assertEqual(result["content"], "line 1\nline 2\nline 3\n")

    def test_read_nonexistent_file(self):
        read = self._get_tool()
        result = read(os.path.join(self.test_dir, "nope.txt"))
        self.assertIn("error", result)

    def test_read_directory_not_file(self):
        read = self._get_tool()
        result = read(self.test_dir)
        self.assertIn("error", result)

    def test_read_empty_file(self):
        empty_file = os.path.join(self.test_dir, "empty.txt")
        with open(empty_file, "w") as f:
            f.write("")
        read = self._get_tool()
        result = read(empty_file)
        self.assertNotIn("error", result)
        self.assertEqual(result["content"], "")


# ═══════════════════════════════════════════════════════════════════
# LIST_FILES TOOL TESTS
# ═══════════════════════════════════════════════════════════════════

class TestListFiles(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with open(os.path.join(self.test_dir, "a.py"), "w") as f:
            f.write("# file a")
        with open(os.path.join(self.test_dir, "b.txt"), "w") as f:
            f.write("file b")
        os.makedirs(os.path.join(self.test_dir, "subdir"))
        with open(os.path.join(self.test_dir, ".hidden"), "w") as f:
            f.write("secret")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _get_tool(self):
        import tools  # noqa: F401
        from tool_registry import get_tool
        return get_tool("list_files").func

    def test_list_directory(self):
        ls = self._get_tool()
        result = ls(self.test_dir)
        names = [f["name"] for f in result["files"]]
        self.assertIn("a.py", names)
        self.assertIn("b.txt", names)
        self.assertIn("subdir", names)

    def test_list_types_correct(self):
        ls = self._get_tool()
        result = ls(self.test_dir)
        by_name = {f["name"]: f["type"] for f in result["files"]}
        self.assertEqual(by_name["a.py"], "file")
        self.assertEqual(by_name["subdir"], "dir")

    def test_hidden_files_skipped(self):
        ls = self._get_tool()
        result = ls(self.test_dir)
        names = [f["name"] for f in result["files"]]
        self.assertNotIn(".hidden", names)

    def test_list_empty_directory(self):
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)
        ls = self._get_tool()
        result = ls(empty_dir)
        self.assertEqual(result["files"], [])

    def test_list_nonexistent_path(self):
        ls = self._get_tool()
        result = ls(os.path.join(self.test_dir, "nope"))
        self.assertIn("error", result)

    def test_list_file_not_dir(self):
        ls = self._get_tool()
        result = ls(os.path.join(self.test_dir, "a.py"))
        self.assertIn("error", result)

    def test_list_sorted_output(self):
        ls = self._get_tool()
        result = ls(self.test_dir)
        names = [f["name"] for f in result["files"]]
        self.assertEqual(names, sorted(names))


# ═══════════════════════════════════════════════════════════════════
# EDIT_FILE TOOL TESTS
# ═══════════════════════════════════════════════════════════════════

class TestEditFile(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _get_tool(self):
        import tools  # noqa: F401
        from tool_registry import get_tool
        return get_tool("edit_file").func

    def test_create_new_file(self):
        edit = self._get_tool()
        filepath = os.path.join(self.test_dir, "new.py")
        result = edit(path=filepath, new_str="print('hello')", old_str="")
        self.assertEqual(result["action"], "created_file")
        with open(filepath) as f:
            self.assertEqual(f.read(), "print('hello')")

    def test_create_file_in_nested_dir(self):
        edit = self._get_tool()
        filepath = os.path.join(self.test_dir, "sub", "deep", "file.txt")
        result = edit(path=filepath, new_str="nested content", old_str="")
        self.assertEqual(result["action"], "created_file")
        self.assertTrue(os.path.exists(filepath))

    def test_overwrite_existing_file(self):
        edit = self._get_tool()
        filepath = os.path.join(self.test_dir, "existing.txt")
        with open(filepath, "w") as f:
            f.write("old content")
        result = edit(path=filepath, new_str="new content", old_str="")
        self.assertEqual(result["action"], "created_file")
        with open(filepath) as f:
            self.assertEqual(f.read(), "new content")

    def test_replace_string(self):
        edit = self._get_tool()
        filepath = os.path.join(self.test_dir, "code.py")
        with open(filepath, "w") as f:
            f.write("def hello():\n    print('hello')\n")
        result = edit(path=filepath, old_str="hello", new_str="world")
        self.assertEqual(result["action"], "edited")
        with open(filepath) as f:
            content = f.read()
        self.assertTrue(content.startswith("def world():"))
        self.assertIn("hello", content)  # Second occurrence untouched

    def test_replace_string_not_found(self):
        edit = self._get_tool()
        filepath = os.path.join(self.test_dir, "code.py")
        with open(filepath, "w") as f:
            f.write("some content here")
        result = edit(path=filepath, old_str="nonexistent", new_str="replacement")
        self.assertEqual(result["action"], "old_str_not_found")
        with open(filepath) as f:
            self.assertEqual(f.read(), "some content here")

    def test_replace_in_nonexistent_file(self):
        edit = self._get_tool()
        filepath = os.path.join(self.test_dir, "nope.py")
        result = edit(path=filepath, old_str="find", new_str="replace")
        self.assertIn("error", result)

    def test_replace_multiline(self):
        edit = self._get_tool()
        filepath = os.path.join(self.test_dir, "multi.py")
        with open(filepath, "w") as f:
            f.write("line1\nline2\nline3\n")
        result = edit(path=filepath, old_str="line1\nline2", new_str="replaced1\nreplaced2")
        self.assertEqual(result["action"], "edited")
        with open(filepath) as f:
            self.assertEqual(f.read(), "replaced1\nreplaced2\nline3\n")

    def test_delete_string(self):
        edit = self._get_tool()
        filepath = os.path.join(self.test_dir, "trim.txt")
        with open(filepath, "w") as f:
            f.write("keep this REMOVE THIS keep this too")
        result = edit(path=filepath, old_str="REMOVE THIS ", new_str="")
        self.assertEqual(result["action"], "edited")
        with open(filepath) as f:
            self.assertEqual(f.read(), "keep this keep this too")


# ═══════════════════════════════════════════════════════════════════
# WEB_SEARCH TOOL TESTS
# ═══════════════════════════════════════════════════════════════════

class TestWebSearch(unittest.TestCase):

    def setUp(self):
        import importlib
        import tools
        for name in list(sys.modules):
            if name.startswith("tools."):
                importlib.reload(sys.modules[name])
        importlib.reload(tools)

    def _get_tool(self):
        from tool_registry import get_tool
        return get_tool("web_search").func

    def test_web_search_registered(self):
        from tool_registry import get_tool_names
        self.assertIn("web_search", get_tool_names())

    def test_empty_query_returns_error(self):
        search = self._get_tool()
        result = search(query="")
        self.assertIn("error", result)
        self.assertIn("Empty", result["error"])

    def test_whitespace_query_returns_error(self):
        search = self._get_tool()
        result = search(query="   ")
        self.assertIn("error", result)

    def test_num_results_clamped_low(self):
        search = self._get_tool()
        result = search(query="python", num_results=-5)
        self.assertIsInstance(result, dict)
        if "error" not in result:
            self.assertLessEqual(result["num_results"], 1)

    def test_num_results_clamped_high(self):
        search = self._get_tool()
        result = search(query="python", num_results=100)
        self.assertIsInstance(result, dict)
        if "error" not in result:
            self.assertLessEqual(result["num_results"], 10)

    def test_result_structure(self):
        search = self._get_tool()
        result = search(query="Python programming language", num_results=2)
        if "error" in result:
            self.assertIsInstance(result["error"], str)
            return
        self.assertIn("query", result)
        self.assertIn("results", result)
        self.assertIn("num_results", result)
        for entry in result["results"]:
            self.assertIn("title", entry)
            self.assertIn("url", entry)
            self.assertIn("snippet", entry)

    def test_search_returns_dict(self):
        search = self._get_tool()
        result = search(query="what is 2+2")
        self.assertIsInstance(result, dict)

    def test_execute_via_registry(self):
        from tool_registry import execute_tool
        result = execute_tool("web_search", {"query": "hello world"})
        self.assertIsInstance(result, dict)
        self.assertTrue("results" in result or "error" in result)

    def test_default_num_results(self):
        search = self._get_tool()
        result = search(query="test query")
        if "error" not in result:
            self.assertLessEqual(result["num_results"], 3)


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION: REGISTRY + TOOLS TOGETHER
# ═══════════════════════════════════════════════════════════════════

class TestRegistryIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        import tools  # noqa: F401

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_execute_read_file(self):
        filepath = os.path.join(self.test_dir, "test.txt")
        with open(filepath, "w") as f:
            f.write("integration test")
        from tool_registry import execute_tool
        result = execute_tool("read_file", {"filename": filepath})
        self.assertEqual(result["content"], "integration test")

    def test_execute_list_files(self):
        with open(os.path.join(self.test_dir, "x.py"), "w") as f:
            f.write("")
        from tool_registry import execute_tool
        result = execute_tool("list_files", {"path": self.test_dir})
        names = [f["name"] for f in result["files"]]
        self.assertIn("x.py", names)

    def test_execute_edit_file_create(self):
        filepath = os.path.join(self.test_dir, "created.txt")
        from tool_registry import execute_tool
        result = execute_tool("edit_file", {
            "path": filepath, "new_str": "created via registry", "old_str": ""
        })
        self.assertEqual(result["action"], "created_file")
        with open(filepath) as f:
            self.assertEqual(f.read(), "created via registry")

    def test_full_workflow_create_read_edit_read(self):
        filepath = os.path.join(self.test_dir, "workflow.py")
        from tool_registry import execute_tool

        result = execute_tool("edit_file", {
            "path": filepath,
            "new_str": "def greet():\n    print('hello')\n",
            "old_str": ""
        })
        self.assertEqual(result["action"], "created_file")

        result = execute_tool("read_file", {"filename": filepath})
        self.assertIn("def greet():", result["content"])

        result = execute_tool("edit_file", {
            "path": filepath, "old_str": "hello", "new_str": "goodbye"
        })
        self.assertEqual(result["action"], "edited")

        result = execute_tool("read_file", {"filename": filepath})
        self.assertIn("goodbye", result["content"])
        self.assertNotIn("hello", result["content"])


# ═══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSystemPrompt(unittest.TestCase):

    def setUp(self):
        import importlib
        import tools
        for name in list(sys.modules):
            if name.startswith("tools."):
                importlib.reload(sys.modules[name])
        importlib.reload(tools)

    def test_full_prompt_contains_personality(self):
        from system_prompt import get_full_system_prompt
        prompt = get_full_system_prompt()
        self.assertIn("LOKI", prompt)
        self.assertIn("JARVIS", prompt)
        self.assertIn("Cheerful but precise", prompt)

    def test_full_prompt_contains_tool_schemas(self):
        from system_prompt import get_full_system_prompt
        prompt = get_full_system_prompt()
        self.assertIn("read_file", prompt)
        self.assertIn("list_files", prompt)
        self.assertIn("edit_file", prompt)
        self.assertIn("web_search", prompt)

    def test_full_prompt_contains_tool_format(self):
        from system_prompt import get_full_system_prompt
        prompt = get_full_system_prompt()
        self.assertIn("tool:", prompt)
        self.assertIn("TOOL_NAME", prompt)
        self.assertIn("tool_result", prompt)

    def test_full_prompt_contains_examples(self):
        from system_prompt import get_full_system_prompt
        prompt = get_full_system_prompt()
        self.assertIn('tool: read_file({"filename": "main.py"})', prompt)

    def test_personality_only(self):
        from system_prompt import get_personality_only
        personality = get_personality_only()
        self.assertIn("LOKI", personality)
        self.assertNotIn("TOOL_NAME", personality)

    def test_full_prompt_contains_honesty_rules(self):
        from system_prompt import get_full_system_prompt
        prompt = get_full_system_prompt()
        self.assertIn("NEVER fake confidence", prompt)

    def test_full_prompt_contains_search_guidance(self):
        from system_prompt import get_full_system_prompt
        prompt = get_full_system_prompt()
        self.assertIn("Let me look that up", prompt)

    def test_prompt_is_string(self):
        from system_prompt import get_full_system_prompt
        prompt = get_full_system_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 500)

    def test_no_think_flag_when_disabled(self):
        """When ALLOW_THINKING is False, prompt should contain /no_think."""
        from unittest.mock import patch
        with patch("config.ALLOW_THINKING", False):
                import importlib
                import system_prompt
                importlib.reload(system_prompt)
                prompt = system_prompt.get_full_system_prompt()
                self.assertIn("/no_think", prompt)

    def test_no_think_flag_absent_when_enabled(self):
        """When ALLOW_THINKING is True, prompt should NOT contain /no_think."""
        from unittest.mock import patch
        with patch("config.ALLOW_THINKING", True):
            import importlib
            import system_prompt
            importlib.reload(system_prompt)
            prompt = system_prompt.get_full_system_prompt()
            self.assertNotIn("/no_think", prompt)

# ═══════════════════════════════════════════════════════════════════
# LLM CLIENT TESTS
# ═══════════════════════════════════════════════════════════════════

class TestLLMClient(unittest.TestCase):

    def test_call_llm_builds_correct_payload(self):
        from unittest.mock import patch, MagicMock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Hello! I'm Loki."}
        }
        with patch("llm.requests.post", return_value=mock_response) as mock_post:
            from llm import call_llm
            result = call_llm(
                messages=[{"role": "user", "content": "Hi"}],
                system_prompt="You are Loki."
            )
            self.assertEqual(result, "Hello! I'm Loki.")
            payload = mock_post.call_args[1]["json"]
            self.assertEqual(payload["model"], "loki")
            self.assertEqual(payload["stream"], False)
            self.assertEqual(payload["system"], "You are Loki.")

    def test_call_llm_filters_system_messages(self):
        from unittest.mock import patch, MagicMock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Got it."}
        }
        with patch("llm.requests.post", return_value=mock_response) as mock_post:
            from llm import call_llm
            call_llm(messages=[
                {"role": "system", "content": "System prompt here"},
                {"role": "user", "content": "Hello"},
            ])
            payload = mock_post.call_args[1]["json"]
            self.assertEqual(payload["system"], "System prompt here")
            self.assertEqual(len(payload["messages"]), 1)

    def test_call_llm_sends_conversation_history(self):
        from unittest.mock import patch, MagicMock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Sure thing."}
        }
        with patch("llm.requests.post", return_value=mock_response) as mock_post:
            from llm import call_llm
            call_llm(messages=[
                {"role": "user", "content": "Create a file"},
                {"role": "assistant", "content": "tool: edit_file({})"},
                {"role": "user", "content": 'tool_result({"action": "created"})'},
            ])
            payload = mock_post.call_args[1]["json"]
            self.assertEqual(len(payload["messages"]), 3)

    def test_call_llm_connection_error(self):
        from unittest.mock import patch
        with patch("llm.requests.post", side_effect=__import__("requests").ConnectionError):
            from llm import call_llm
            with self.assertRaises(ConnectionError) as ctx:
                call_llm(messages=[{"role": "user", "content": "Hi"}])
            self.assertIn("Cannot connect", str(ctx.exception))

    def test_call_llm_timeout_error(self):
        from unittest.mock import patch
        with patch("llm.requests.post", side_effect=__import__("requests").Timeout):
            from llm import call_llm
            with self.assertRaises(RuntimeError) as ctx:
                call_llm(messages=[{"role": "user", "content": "Hi"}])
            self.assertIn("timed out", str(ctx.exception))

    def test_call_llm_api_error_status(self):
        from unittest.mock import patch, MagicMock
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        with patch("llm.requests.post", return_value=mock_response):
            from llm import call_llm
            with self.assertRaises(RuntimeError) as ctx:
                call_llm(messages=[{"role": "user", "content": "Hi"}])
            self.assertIn("500", str(ctx.exception))

    def test_call_llm_empty_response(self):
        from unittest.mock import patch, MagicMock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": ""}
        }
        with patch("llm.requests.post", return_value=mock_response):
            from llm import call_llm
            with self.assertRaises(RuntimeError) as ctx:
                call_llm(messages=[{"role": "user", "content": "Hi"}])
            self.assertIn("empty response", str(ctx.exception))

    def test_call_llm_includes_model_options(self):
        from unittest.mock import patch, MagicMock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Ok."}
        }
        with patch("llm.requests.post", return_value=mock_response) as mock_post:
            from llm import call_llm
            call_llm(messages=[{"role": "user", "content": "Hi"}])
            payload = mock_post.call_args[1]["json"]
            self.assertIn("options", payload)
            self.assertEqual(payload["options"]["temperature"], 0.7)
            self.assertEqual(payload["options"]["top_p"], 0.9)
            self.assertEqual(payload["options"]["num_ctx"], 4096)

    def test_check_connection_ollama_down(self):
        from unittest.mock import patch
        with patch("llm.requests.get", side_effect=__import__("requests").ConnectionError):
            from llm import check_ollama_connection
            result = check_ollama_connection()
            self.assertEqual(result["status"], "error")

    def test_check_connection_model_found(self):
        from unittest.mock import patch, MagicMock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "loki:latest"}, {"name": "qwen3:8b"}]
        }
        with patch("llm.requests.get", return_value=mock_response):
            from llm import check_ollama_connection
            result = check_ollama_connection()
            self.assertEqual(result["status"], "ok")

    def test_check_connection_model_missing(self):
        from unittest.mock import patch, MagicMock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3:latest"}]
        }
        with patch("llm.requests.get", return_value=mock_response):
            from llm import check_ollama_connection
            result = check_ollama_connection()
            self.assertEqual(result["status"], "error")
            self.assertIn("not found", result["message"])


# ═══════════════════════════════════════════════════════════════════
# SPINNER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSpinner(unittest.TestCase):

    def test_spinner_starts_and_stops(self):
        from spinner import Spinner
        s = Spinner("thinking")
        s.start()
        self.assertTrue(s._running)
        time.sleep(0.2)
        s.stop()
        self.assertFalse(s._running)

    def test_spinner_context_manager(self):
        from spinner import Spinner
        with Spinner("thinking") as s:
            self.assertTrue(s._running)
            time.sleep(0.2)
        self.assertFalse(s._running)

    def test_spinner_context_manager_on_exception(self):
        from spinner import Spinner
        s = None
        try:
            with Spinner("tool", tool_name="read_file") as s:
                raise ValueError("Test exception")
        except ValueError:
            pass
        self.assertFalse(s._running)

    def test_spinner_double_start(self):
        from spinner import Spinner
        s = Spinner("thinking")
        s.start()
        thread1 = s._thread
        s.start()
        thread2 = s._thread
        self.assertIs(thread1, thread2)
        s.stop()

    def test_spinner_double_stop(self):
        from spinner import Spinner
        s = Spinner("thinking")
        s.start()
        time.sleep(0.1)
        s.stop()
        s.stop()
        self.assertFalse(s._running)

    def test_spinner_stop_without_start(self):
        from spinner import Spinner
        s = Spinner("thinking")
        s.stop()
        self.assertFalse(s._running)

    def test_thinking_message_selection(self):
        from spinner import _get_message, THINKING_MESSAGES
        msg = _get_message("thinking")
        self.assertIn(msg, THINKING_MESSAGES)

    def test_tool_message_selection_known_tool(self):
        from spinner import _get_message, TOOL_MESSAGES
        msg = _get_message("tool", "web_search")
        self.assertIn(msg, TOOL_MESSAGES["web_search"])

    def test_tool_message_selection_unknown_tool(self):
        from spinner import _get_message, TOOL_MESSAGES
        msg = _get_message("tool", "nonexistent_tool")
        self.assertIn(msg, TOOL_MESSAGES["_default"])

    def test_tool_message_for_each_tool(self):
        from spinner import TOOL_MESSAGES
        for tool_name in ["read_file", "list_files", "edit_file", "web_search"]:
            self.assertIn(tool_name, TOOL_MESSAGES)
            self.assertGreater(len(TOOL_MESSAGES[tool_name]), 0)

    def test_spinner_mode_sets_initial_message(self):
        from spinner import Spinner, THINKING_MESSAGES, TOOL_MESSAGES
        s1 = Spinner("thinking")
        self.assertIn(s1._message, THINKING_MESSAGES)
        s2 = Spinner("tool", tool_name="edit_file")
        self.assertIn(s2._message, TOOL_MESSAGES["edit_file"])

    def test_spinner_frames_exist(self):
        from spinner import SPINNER_FRAMES
        self.assertGreater(len(SPINNER_FRAMES), 0)

    def test_spinner_completes_quickly(self):
        from spinner import Spinner
        s = Spinner("thinking")
        s.start()
        time.sleep(0.2)
        start = time.time()
        s.stop()
        elapsed = time.time() - start
        self.assertLess(elapsed, 1.0)


# ═══════════════════════════════════════════════════════════════════
# AGENT LOOP TESTS
# ═══════════════════════════════════════════════════════════════════

class TestAgentLoop(unittest.TestCase):

    def setUp(self):
        import importlib
        import tools as t
        for name in list(sys.modules):
            if name.startswith("tools."):
                importlib.reload(sys.modules[name])
        importlib.reload(t)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_preflight_check_success(self):
        from unittest.mock import patch
        from loki import preflight_check
        with patch("loki.check_ollama_connection", return_value={
            "status": "ok", "message": "All good"
        }):
            self.assertTrue(preflight_check())

    def test_preflight_check_failure(self):
        from unittest.mock import patch
        from loki import preflight_check
        with patch("loki.check_ollama_connection", return_value={
            "status": "error", "message": "Cannot connect"
        }):
            self.assertFalse(preflight_check())

    def test_print_tool_call_short_args(self):
        from loki import print_tool_call
        print_tool_call("read_file", {"filename": "test.py"})

    def test_print_tool_call_long_args(self):
        from loki import print_tool_call
        print_tool_call("edit_file", {"path": "test.py", "new_str": "x" * 500, "old_str": ""})

    def test_print_tool_result_success(self):
        from loki import print_tool_result
        print_tool_result("read_file", {"file_path": "/tmp/test.py", "content": "hello"})

    def test_print_tool_result_error(self):
        from loki import print_tool_result
        print_tool_result("read_file", {"error": "File not found"})

    def test_full_conversation_no_tools(self):
        from unittest.mock import patch, MagicMock
        import loki
        with patch.object(loki, "call_llm", return_value="Hello! I'm Loki."):
            conversation = []
            conversation.append({"role": "user", "content": "Hi"})
            response = loki.call_llm(messages=conversation, system_prompt="test")
            invocations = loki.extract_tool_invocations(response)
            self.assertEqual(invocations, [])
            self.assertEqual(response, "Hello! I'm Loki.")

    def test_full_conversation_with_tool_call(self):
        from unittest.mock import patch, MagicMock
        import loki

        filepath = os.path.join(self.test_dir, "demo.py")
        with open(filepath, "w") as f:
            f.write("print('hello')")

        llm_responses = [
            f'Let me read that.\ntool: read_file({{"filename": "{filepath}"}})',
            "The file contains a simple print statement."
        ]
        call_count = [0]

        def mock_call_llm(messages, system_prompt=None):
            idx = min(call_count[0], len(llm_responses) - 1)
            call_count[0] += 1
            return llm_responses[idx]

        with patch.object(loki, "call_llm", side_effect=mock_call_llm):
            conversation = []
            conversation.append({"role": "user", "content": "Read demo.py"})

            response = loki.call_llm(messages=conversation, system_prompt="test")
            invocations = loki.extract_tool_invocations(response)
            self.assertEqual(len(invocations), 1)

            conversation.append({"role": "assistant", "content": response})
            name, args = invocations[0]
            result = loki.execute_tool(name, args)
            self.assertEqual(result["content"], "print('hello')")

            conversation.append({"role": "user", "content": f"tool_result({json.dumps(result)})"})
            response = loki.call_llm(messages=conversation, system_prompt="test")
            invocations = loki.extract_tool_invocations(response)
            self.assertEqual(invocations, [])

    def test_tool_round_safety_limit(self):
        import loki, inspect
        source = inspect.getsource(loki.run_agent_loop)
        self.assertIn("max_tool_rounds", source)

    def test_meta_command_quit(self):
        import loki, inspect
        source = inspect.getsource(loki.run_agent_loop)
        self.assertIn('"quit"', source)
        self.assertIn('"exit"', source)

    def test_banner_exists(self):
        from loki import BANNER
        self.assertIn("L O K I", BANNER)

    def test_tool_result_injected_as_user_message(self):
        conversation = []
        result = {"file_path": "/tmp/x.py", "content": "hello"}
        conversation.append({
            "role": "user",
            "content": f"tool_result({json.dumps(result)})"
        })
        self.assertEqual(conversation[0]["role"], "user")
        self.assertIn("tool_result", conversation[0]["content"])


# ═══════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)

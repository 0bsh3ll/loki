"""
LOKI Phase 1 Tests

Run: python -m tests.test_tools  (from loki/ directory)
  or: python tests/test_tools.py

Tests every component in isolation — no LLM needed.
"""

import os
import sys
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


# ═══════════════════════════════════════════════════════════════════
# TOOL REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════

class TestToolRegistry(unittest.TestCase):
    """Tests for the decorator-based tool registry."""

    def setUp(self):
        """Clear the registry before each test so they're independent."""
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
        """Generated schema should include the tool name."""

        @register_tool("finder")
        def finder(path: str) -> dict:
            """Finds things."""
            return {}

        schema = get_all_tool_schemas()
        self.assertIn("finder", schema)

    def test_schema_contains_description(self):
        """Generated schema should include the tool description."""

        @register_tool("searcher")
        def searcher(query: str) -> dict:
            """Searches the web for information."""
            return {}

        schema = get_all_tool_schemas()
        self.assertIn("Searches the web for information", schema)

    def test_schema_contains_parameter_info(self):
        """Generated schema should list parameters with types and defaults."""

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
        """Schema for empty registry returns a placeholder string."""
        schema = get_all_tool_schemas()
        self.assertIn("No tools registered", schema)

    def test_schema_multiple_tools(self):
        """Schema includes all registered tools separated by delimiters."""

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
        # Should have a separator between them
        self.assertIn("=" * 40, schema)

    # ─── Execution ─────────────────────────────────────────────────

    def test_execute_tool_success(self):
        """execute_tool should call the function and return its result."""

        @register_tool("adder")
        def adder(a: int, b: int) -> dict:
            """Adds two numbers."""
            return {"sum": a + b}

        result = execute_tool("adder", {"a": 3, "b": 7})
        self.assertEqual(result, {"sum": 10})

    def test_execute_tool_unknown(self):
        """Executing an unknown tool returns an error dict."""
        result = execute_tool("nonexistent", {})
        self.assertIn("error", result)
        self.assertIn("Unknown tool", result["error"])

    def test_execute_tool_bad_args(self):
        """Wrong arguments return an error dict, not a crash."""

        @register_tool("strict")
        def strict(required_arg: str) -> dict:
            """Needs an argument."""
            return {"got": required_arg}

        result = execute_tool("strict", {"wrong_arg": "oops"})
        self.assertIn("error", result)
        self.assertIn("failed", result["error"])

    def test_execute_tool_exception_handling(self):
        """If the tool function raises, we get an error dict."""

        @register_tool("crasher")
        def crasher() -> dict:
            """Will crash."""
            raise RuntimeError("Boom!")

        result = execute_tool("crasher", {})
        self.assertIn("error", result)
        self.assertIn("RuntimeError", result["error"])
        self.assertIn("Boom!", result["error"])

    def test_execute_tool_with_defaults(self):
        """Tools with default parameters should work with partial args."""

        @register_tool("greeter")
        def greeter(name: str, shout: bool = False) -> dict:
            """Greets someone."""
            msg = f"Hello, {name}!"
            if shout:
                msg = msg.upper()
            return {"message": msg}

        # Without optional arg
        result = execute_tool("greeter", {"name": "Loki"})
        self.assertEqual(result, {"message": "Hello, Loki!"})

        # With optional arg
        result = execute_tool("greeter", {"name": "Loki", "shout": True})
        self.assertEqual(result, {"message": "HELLO, LOKI!"})


# ═══════════════════════════════════════════════════════════════════
# TOOL PARSER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestToolParser(unittest.TestCase):
    """Tests for parsing tool calls from LLM text output."""

    # ─── Single Call ───────────────────────────────────────────────

    def test_single_tool_call(self):
        """Parse a single well-formed tool call."""
        text = 'tool: read_file({"filename": "main.py"})'
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "read_file")
        self.assertEqual(result[0][1], {"filename": "main.py"})

    def test_single_call_with_multiple_args(self):
        """Parse a tool call with multiple arguments."""
        text = 'tool: edit_file({"path": "x.py", "old_str": "foo", "new_str": "bar"})'
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)
        name, args = result[0]
        self.assertEqual(name, "edit_file")
        self.assertEqual(args["path"], "x.py")
        self.assertEqual(args["old_str"], "foo")
        self.assertEqual(args["new_str"], "bar")

    def test_single_call_with_integer_arg(self):
        """Numeric arguments should parse correctly."""
        text = 'tool: web_search({"query": "python", "num_results": 5})'
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1]["num_results"], 5)

    # ─── Multiple Calls ───────────────────────────────────────────

    def test_multiple_tool_calls(self):
        """Parse multiple tool calls on separate lines."""
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
        """Regular text with no tool calls returns empty list."""
        text = "Sure, I can help you with that. Let me explain..."
        result = extract_tool_invocations(text)
        self.assertEqual(result, [])

    def test_no_tool_calls_empty_string(self):
        """Empty string returns empty list."""
        result = extract_tool_invocations("")
        self.assertEqual(result, [])

    # ─── Mixed Content ─────────────────────────────────────────────

    def test_mixed_prose_and_tool_call(self):
        """Tool calls mixed with regular text — only tool lines extracted."""
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
        """Broken JSON is skipped, no crash."""
        text = "tool: read_file({broken json here})"
        result = extract_tool_invocations(text)
        self.assertEqual(result, [])

    def test_missing_closing_paren(self):
        """Missing closing parenthesis is skipped."""
        text = 'tool: read_file({"filename": "x.py"}'
        result = extract_tool_invocations(text)
        self.assertEqual(result, [])

    def test_no_parentheses(self):
        """Tool name without parentheses is skipped."""
        text = "tool: read_file"
        result = extract_tool_invocations(text)
        self.assertEqual(result, [])

    def test_non_dict_json(self):
        """JSON that's not a dict (e.g. a list) is skipped."""
        text = 'tool: read_file(["not", "a", "dict"])'
        result = extract_tool_invocations(text)
        self.assertEqual(result, [])

    def test_empty_args(self):
        """Empty dict args should parse fine."""
        text = "tool: list_files({})"
        # This is technically valid — a tool call with no args
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("list_files", {}))

    # ─── Whitespace Handling ───────────────────────────────────────

    def test_leading_whitespace(self):
        """Lines with leading whitespace should still parse."""
        text = '   tool: read_file({"filename": "x.py"})'
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)

    def test_spaces_around_tool_name(self):
        """Extra spaces around the tool name should be handled."""
        text = 'tool:   read_file  ({"filename": "x.py"})'
        result = extract_tool_invocations(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "read_file")

    # ─── has_tool_calls Convenience ────────────────────────────────

    def test_has_tool_calls_true(self):
        text = 'tool: read_file({"filename": "x.py"})'
        self.assertTrue(has_tool_calls(text))

    def test_has_tool_calls_false(self):
        text = "Just a normal response."
        self.assertFalse(has_tool_calls(text))

    # ─── strip_tool_lines ──────────────────────────────────────────

    def test_strip_tool_lines_removes_calls(self):
        """strip_tool_lines should return only the prose."""
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
        """If the entire response is tool calls, result should be empty."""
        text = (
            'tool: read_file({"filename": "a.py"})\n'
            'tool: read_file({"filename": "b.py"})'
        )
        stripped = strip_tool_lines(text)
        self.assertEqual(stripped, "")

    def test_strip_tool_lines_no_tools(self):
        """No tool lines means the full text is returned."""
        text = "Hello, I'm Loki."
        stripped = strip_tool_lines(text)
        self.assertEqual(stripped, text)


# ═══════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run with verbose output so each test name is visible
    unittest.main(verbosity=2)

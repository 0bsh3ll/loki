#!/usr/bin/env python3
"""
LOKI — Local AI Assistant

The main entry point. Wires together every module into a
CLI agent loop:

    system_prompt  → personality + tools
    llm            → talks to Ollama
    tool_registry  → discovers and executes tools
    tool_parser    → detects tool calls in LLM output
    spinner        → animated status while waiting

Usage:
    1. ollama create loki -f Modelfile
    2. ollama serve
    3. python loki.py

That's it. You're talking to Loki.
"""

import json
import re
import sys

# ─── Import all modules ────────────────────────────────────────────
from config import (
    COLOR_USER, COLOR_ASSISTANT, COLOR_TOOL,
    COLOR_ERROR, COLOR_THINKING, COLOR_RESET, COLOR_BOLD,
)
from system_prompt import get_full_system_prompt
from llm import call_llm, check_ollama_connection
from tool_registry import execute_tool
from tool_parser import extract_tool_invocations, strip_think_tags
from spinner import Spinner

# Trigger auto-discovery of all tools
import tools  # noqa: F401


# ─── Startup Banner ────────────────────────────────────────────────

BANNER = f"""
{COLOR_BOLD}             .
            / \\
   |\\      /   \\      /|
   \\ \\    /     \\    / /
    \\ \\  /       \\  / /
     \\ \\/    _    \\/ /    ╔══════════════════════════════════╗
      \\__   / \\   __/     ║           L O K I                ║
         |  \\_/  |        ║   Calculated. Precise. AI.       ║
         \\_______/        ╚══════════════════════════════════╝{COLOR_RESET}

{COLOR_THINKING}Type your message and press Enter.
Type 'quit' or 'exit' to leave.
Type 'tools' to list available tools.
Type 'clear' to reset conversation history.{COLOR_RESET}
"""

# Patterns that indicate the model should have called a tool but didn't —
# either it announced one and never emitted the call, or it refused by
# claiming it has no tools. When matched with zero tool invocations, we
# nudge the model once per user turn.
_ANNOUNCES_TOOL = re.compile(
    r"\b(let me (look|search|check|find|pull)|"
    r"i(?:'ll| will) (search|look|check|find|pull)|"
    r"looking (it|that) up|"
    r"i will search|"
    r"please wait while)\b",
    re.IGNORECASE,
)

_REFUSES_TOOL = re.compile(
    r"(i (?:don'?t|do not) have (?:access|a (?:real[- ]?time )?search|a (?:live )?search|a browsing|internet|web)|"
    r"i (?:cannot|can'?t) (?:search|browse|access the (?:internet|web)|look (?:it|that) up|check online)|"
    r"i (?:don'?t|do not) have (?:the )?ability to (?:search|browse|access)|"
    r"(?:my|the) (?:internal )?knowledge (?:cutoff|is limited)|"
    r"without a (?:real[- ]?time|live) search tool|"
    r"i am not able to (?:search|browse|access))",
    re.IGNORECASE,
)

# Triggers in the USER's question that require web_search before any answer.
# If the user asks for current/time-sensitive info and the model answers
# without calling the tool, we treat it as a hallucination and force a search.
_REQUIRES_SEARCH = re.compile(
    r"\b(latest|current|today|now|recently?|newest|"
    r"as of|this year|right now|"
    r"what(?:'s| is) (?:happening|new)|"
    r"who (?:is|won|wins)|"
    r"weather|price of|score|news)\b",
    re.IGNORECASE,
)


# ─── Streaming Printer ─────────────────────────────────────────────
# Char-level stream display. Suppresses any line that begins with
# ``tool:``, swallows ``<think>...</think>`` spans, and otherwise prints
# tokens to stdout the moment they arrive. The "Loki:" prefix is emitted
# lazily on the first visible char so a tool-call-only reply does not
# leave a dangling label in the terminal.

class StreamPrinter:
    _NOISE = set(" \t-*>`")
    _PREFIX_LIMIT = 5      # how many non-noise chars of a new line we
                           # peek at before committing to print or suppress.
    _DEFER_LT_LIMIT = 10   # if a line starts with "<", wait for ">" or
                           # this many chars before deciding (think-tag guard).

    def __init__(self, prefix: str = "", out=None):
        self._prefix = prefix
        self._prefix_emitted = False
        self._out = out if out is not None else sys.stdout
        self._line_buf = ""
        self._mode = "prefix-check"   # | "streaming" | "suppressing"
        self._in_think = False
        self._tail = ""               # rolling 9-char window for <think>/</think>
        self.printed_anything = False

    # ── output helpers ─────────────────────────────────────────
    def _emit(self, s: str):
        if not s:
            return
        if not self._prefix_emitted and self._prefix:
            self._out.write(self._prefix)
            self._prefix_emitted = True
        self._out.write(s)
        self._out.flush()
        self.printed_anything = True

    def _stripped_prefix(self) -> str:
        i = 0
        while i < len(self._line_buf) and self._line_buf[i] in self._NOISE:
            i += 1
        return self._line_buf[i:]

    # ── decision & line termination ────────────────────────────
    def _decide(self):
        """In prefix-check mode: decide whether the line so far is a
        tool call (suppress) or prose (flush + stream the rest)."""
        stripped = self._stripped_prefix()
        if not stripped:
            return  # only noise so far

        # If line might still be a "<think>" opener, hold off.
        if stripped.startswith("<") and ">" not in stripped \
                and len(stripped) < self._DEFER_LT_LIMIT:
            return

        if (len(stripped) >= self._PREFIX_LIMIT) or (":" in stripped):
            if stripped.lower().startswith("tool:"):
                self._mode = "suppressing"
                self._line_buf = ""
            else:
                self._emit(self._line_buf)
                self._line_buf = ""
                self._mode = "streaming"

    def _newline(self):
        if self._mode == "suppressing":
            self._line_buf = ""
        elif self._mode == "streaming":
            self._emit("\n")
        else:  # prefix-check, line ended undecided
            stripped = self._stripped_prefix()
            if stripped.lower().startswith("tool:"):
                pass  # full line was a tool call — drop
            else:
                self._emit(self._line_buf + "\n")
            self._line_buf = ""
        self._mode = "prefix-check"

    # ── main entry points ──────────────────────────────────────
    def feed(self, text: str):
        for ch in text:
            self._tail = (self._tail + ch)[-9:]

            # Think-tag handling first (may straddle chunk boundaries).
            if not self._in_think and self._tail.endswith("<think>"):
                self._in_think = True
                # If the opener was still in our prefix-check buffer,
                # retract it so we don't print "<think" mid-stream.
                if self._line_buf.endswith("<think"):
                    self._line_buf = self._line_buf[:-6]
                continue
            if self._in_think:
                if self._tail.endswith("</think>"):
                    self._in_think = False
                continue

            if ch == "\n":
                self._newline()
                continue

            if self._mode == "streaming":
                self._emit(ch)
                continue
            if self._mode == "suppressing":
                continue

            # prefix-check mode
            self._line_buf += ch
            self._decide()

    def flush(self):
        # Drain any pending line as if a newline had arrived.
        if self._line_buf or self._mode == "streaming":
            self._newline()


def print_tool_call(name: str, args: dict):
    """Display a tool call in the terminal. web_search gets a louder banner
    so it is obvious to the user when Loki reaches the internet."""
    if name == "web_search":
        query = args.get("query", "")
        print(
            f"{COLOR_TOOL}{COLOR_BOLD}  [ web_search ] "
            f"{COLOR_RESET}{COLOR_TOOL}searching the web for: "
            f"{COLOR_BOLD}\"{query}\"{COLOR_RESET}"
        )
        return

    args_short = json.dumps(args)
    if len(args_short) > 80:
        args_short = args_short[:77] + "..."
    print(f"{COLOR_TOOL}  ↳ {name}({args_short}){COLOR_RESET}")


def print_tool_result(name: str, result: dict):
    """Display a tool result summary in the terminal."""
    if "error" in result:
        print(f"{COLOR_ERROR}  ✗ {name} failed: {result['error']}{COLOR_RESET}")
        return

    if name == "web_search":
        n = result.get("num_results", len(result.get("results", [])))
        print(
            f"{COLOR_TOOL}{COLOR_BOLD}  [ web_search ] "
            f"{COLOR_RESET}{COLOR_TOOL}got {n} result"
            f"{'s' if n != 1 else ''}{COLOR_RESET}"
        )
        for i, r in enumerate(result.get("results", []), 1):
            title = (r.get("title") or "")[:90]
            url = r.get("url") or ""
            print(f"{COLOR_TOOL}     {i}. {title}{COLOR_RESET}")
            if url:
                print(f"{COLOR_THINKING}        {url}{COLOR_RESET}")
        return

    # Show a compact summary for everything else
    summary = json.dumps(result)
    if len(summary) > 120:
        summary = summary[:117] + "..."
    print(f"{COLOR_TOOL}  ✓ {name} → {summary}{COLOR_RESET}")


def format_tool_result_for_llm(name: str, result: dict) -> str:
    """
    Build the string that gets fed back to the model as a user message.
    For web_search we ditch raw JSON in favor of a numbered text block —
    small models (qwen2.5-7B, mistral-7B, etc.) ground much better on
    prose than on key/value blobs.
    """
    if name == "web_search" and "error" not in result and result.get("results"):
        query = result.get("query", "")
        lines = [f'Search results for "{query}":', ""]
        for i, r in enumerate(result["results"], 1):
            title = r.get("title", "").strip() or "(no title)"
            url = r.get("url", "").strip()
            snippet = r.get("snippet", "").strip()
            lines.append(f"[{i}] {title}")
            if url:
                lines.append(f"    {url}")
            if snippet:
                lines.append(f"    {snippet}")
            lines.append("")
        lines.append(
            "Answer the user using ONLY the information in the snippets above. "
            "If the answer is not in the snippets, say so — do not fall back on prior knowledge."
        )
        return "\n".join(lines)

    return f"tool_result({json.dumps(result)})"


def preflight_check():
    """
    Verify Ollama is running and the model exists before starting.
    Returns True if everything is good, False otherwise.
    """
    print(f"{COLOR_THINKING}Checking connection to Ollama...{COLOR_RESET}")
    status = check_ollama_connection()

    if status["status"] == "ok":
        print(f"{COLOR_TOOL}  ✓ {status['message']}{COLOR_RESET}\n")
        return True
    else:
        print(f"{COLOR_ERROR}  ✗ {status['message']}{COLOR_RESET}\n")
        return False


def run_agent_loop():
    """
    The main agent loop.

    Outer loop: read user input
    Inner loop: call LLM → parse for tools → execute → repeat
                until LLM responds without tool calls
    """
    # ── Build conversation with system prompt ──────────────────
    system_prompt = get_full_system_prompt()
    conversation = []

    print(BANNER)

    # ── Preflight ──────────────────────────────────────────────
    if not preflight_check():
        print(f"{COLOR_ERROR}Fix the issues above and try again.{COLOR_RESET}")
        sys.exit(1)

    # ── Main loop ──────────────────────────────────────────────
    while True:
        # Get user input
        try:
            user_input = input(f"{COLOR_USER}{COLOR_BOLD}You:{COLOR_RESET} ")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{COLOR_THINKING}Goodbye!{COLOR_RESET}")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # ── Meta commands ──────────────────────────────────────
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{COLOR_THINKING}Goodbye!{COLOR_RESET}")
            break

        if user_input.lower() in ("clear", "/clear", "reset"):
            conversation.clear()
            print(f"{COLOR_TOOL}  ✓ Conversation history cleared.{COLOR_RESET}")
            continue

        if user_input.lower() == "tools":
            from tool_registry import get_all_tools
            all_tools = get_all_tools()
            print(f"\n{COLOR_TOOL}Available tools ({len(all_tools)}):{COLOR_RESET}")
            for name, entry in all_tools.items():
                print(f"  • {COLOR_BOLD}{name}{COLOR_RESET} — {entry.description}")
            print()
            continue

        # ── Add user message to conversation ───────────────────
        conversation.append({
            "role": "user",
            "content": user_input
        })

        # ── Inner loop: LLM ↔ Tool cycle ──────────────────────
        max_tool_rounds = 10  # Safety limit to prevent infinite loops
        tool_round = 0
        nudged_this_turn = False  # Allow at most one "you announced a tool but didn't call it" nudge per user turn

        while tool_round < max_tool_rounds:
            tool_round += 1

            # Call the LLM with streaming + spinner-stops-on-first-token.
            spinner = Spinner("thinking")
            spinner.__enter__()
            spinner_stopped = [False]
            printer = StreamPrinter(
                prefix=f"{COLOR_ASSISTANT}{COLOR_BOLD}Loki:{COLOR_RESET} "
            )

            def _on_chunk(text: str, _spinner=spinner,
                          _stopped=spinner_stopped, _printer=printer):
                if not _stopped[0]:
                    _spinner.__exit__(None, None, None)
                    _stopped[0] = True
                _printer.feed(text)

            try:
                raw = call_llm(
                    messages=conversation,
                    system_prompt=system_prompt,
                    stream=True,
                    on_chunk=_on_chunk,
                )
            except (ConnectionError, RuntimeError) as e:
                if not spinner_stopped[0]:
                    spinner.__exit__(None, None, None)
                printer.flush()
                print(f"{COLOR_ERROR}Error: {e}{COLOR_RESET}")
                # Remove the last user message so they can retry
                conversation.pop()
                break

            # Always close out spinner + line, even on success.
            if not spinner_stopped[0]:
                spinner.__exit__(None, None, None)
            printer.flush()

            assistant_response = strip_think_tags(raw)

            # Check for tool calls
            tool_invocations = extract_tool_invocations(assistant_response)

            if not tool_invocations:
                # ── No tools found ─────────────────────────────
                # Three failure modes we want to rescue before giving up:
                #   1. announce-but-don't-call ("Let me look that up...")
                #   2. false refusal ("I don't have access to a search tool")
                #   3. silent stale-fact answer for a query that clearly
                #      asks about current information ("latest", "today", ...)
                announces = bool(_ANNOUNCES_TOOL.search(assistant_response))
                refuses = bool(_REFUSES_TOOL.search(assistant_response))
                stale_fact = (
                    bool(_REQUIRES_SEARCH.search(user_input))
                    and tool_round == 1  # only on the first answer for this turn
                )

                if not nudged_this_turn and (announces or refuses or stale_fact):
                    conversation.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    if refuses:
                        nudge = (
                            "You DO have tools available right now — see the "
                            "TOOLS section of your system prompt. web_search "
                            "is real and working. Do not claim you lack a "
                            "search tool. Call it now with exactly:\n"
                            "tool: web_search({\"query\": \"...\"})\n"
                            "on its own line, and nothing else."
                        )
                    elif announces:
                        nudge = (
                            "You announced a tool action but emitted no "
                            "tool: NAME({\"key\": \"value\"}) line. If you "
                            "need a tool, call it now on its own line with "
                            "no other text. If you don't, just answer the "
                            "question directly."
                        )
                    else:
                        # stale-fact case
                        nudge = (
                            "STOP. You answered from memory but the user's "
                            "question is about current information. Your "
                            "training data is stale. You do not actually "
                            "know what is current.\n"
                            "Call web_search NOW with exactly:\n"
                            "tool: web_search({\"query\": \"...\"})\n"
                            "on its own line, with nothing else. Then answer "
                            "ONLY from the snippets it returns."
                        )
                    conversation.append({
                        "role": "user",
                        "content": nudge
                    })
                    nudged_this_turn = True
                    continue

                # Regular response — already streamed to the user.
                # Just record it in the conversation history.
                conversation.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                break

            # ── Execute tool calls ─────────────────────────────
            # The streaming printer already showed any prose to the user
            # and suppressed the tool: lines. Just record the full reply
            # (tool lines included, since the parser still reads them).

            # Add the full assistant response (with tool lines) to history
            conversation.append({
                "role": "assistant",
                "content": assistant_response
            })

            for name, args in tool_invocations:
                print_tool_call(name, args)

                # Execute with spinner
                with Spinner("tool", tool_name=name):
                    result = execute_tool(name, args)

                print_tool_result(name, result)

                # Feed the result back to the LLM
                conversation.append({
                    "role": "user",
                    "content": format_tool_result_for_llm(name, result)
                })

            # Loop back — LLM will see the tool results and continue

        else:
            # Hit the safety limit
            print(f"{COLOR_ERROR}Stopped after {max_tool_rounds} tool rounds "
                  f"(safety limit).{COLOR_RESET}")


# ─── Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    run_agent_loop()

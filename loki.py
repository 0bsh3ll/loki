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
import sys

# ─── Import all modules ────────────────────────────────────────────
from config import (
    COLOR_USER, COLOR_ASSISTANT, COLOR_TOOL,
    COLOR_ERROR, COLOR_THINKING, COLOR_RESET, COLOR_BOLD,
)
from system_prompt import get_full_system_prompt
from llm import call_llm, check_ollama_connection
from tool_registry import execute_tool
from tool_parser import extract_tool_invocations, strip_tool_lines,strip_think_tags
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
Type 'tools' to list available tools.{COLOR_RESET}
"""

def print_tool_call(name: str, args: dict):
    """Display a tool call in the terminal."""
    args_short = json.dumps(args)
    if len(args_short) > 80:
        args_short = args_short[:77] + "..."
    print(f"{COLOR_TOOL}  ↳ {name}({args_short}){COLOR_RESET}")


def print_tool_result(name: str, result: dict):
    """Display a tool result summary in the terminal."""
    if "error" in result:
        print(f"{COLOR_ERROR}  ✗ {name} failed: {result['error']}{COLOR_RESET}")
    else:
        # Show a compact summary, not the full result
        summary = json.dumps(result)
        if len(summary) > 120:
            summary = summary[:117] + "..."
        print(f"{COLOR_TOOL}  ✓ {name} → {summary}{COLOR_RESET}")


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

        while tool_round < max_tool_rounds:
            tool_round += 1

            # Call the LLM with spinner
            try:
                with Spinner("thinking"):
                    assistant_response = call_llm(
                        messages=conversation,
                        system_prompt=system_prompt
                    )
                assistant_response = strip_think_tags(assistant_response)
            except (ConnectionError, RuntimeError) as e:
                print(f"{COLOR_ERROR}Error: {e}{COLOR_RESET}")
                # Remove the last user message so they can retry
                conversation.pop()
                break

            # Check for tool calls
            tool_invocations = extract_tool_invocations(assistant_response)

            if not tool_invocations:
                # ── No tools — just a regular response ─────────
                # Strip any accidental tool-like lines and print
                response_text = assistant_response.strip()
                print(f"{COLOR_ASSISTANT}{COLOR_BOLD}Loki:{COLOR_RESET} {response_text}")
                conversation.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                break

            # ── Execute tool calls ─────────────────────────────
            # If the LLM also wrote some prose before the tool call,
            # print that first
            prose = strip_tool_lines(assistant_response)
            if prose:
                print(f"{COLOR_ASSISTANT}{COLOR_BOLD}Loki:{COLOR_RESET} {prose}")

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
                    "content": f"tool_result({json.dumps(result)})"
                })

            # Loop back — LLM will see the tool results and continue

        else:
            # Hit the safety limit
            print(f"{COLOR_ERROR}Stopped after {max_tool_rounds} tool rounds "
                  f"(safety limit).{COLOR_RESET}")


# ─── Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    run_agent_loop()

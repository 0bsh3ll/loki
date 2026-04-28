"""
Non-interactive smoke test: drives one full search round-trip
through the real loki harness against the running Ollama model.
Prints every step so we can see if the model:
  1. emits a valid tool: web_search(...) line,
  2. uses the returned snippets faithfully in its final reply.
"""
import json
import sys

from system_prompt import get_full_system_prompt
from llm import call_llm, check_ollama_connection
from tool_registry import execute_tool
from tool_parser import extract_tool_invocations, strip_think_tags
from loki import (
    _ANNOUNCES_TOOL,
    _REFUSES_TOOL,
    _REQUIRES_SEARCH,
    format_tool_result_for_llm,
)

import tools  # noqa: F401  trigger registry


def run(query: str) -> int:
    status = check_ollama_connection()
    if status["status"] != "ok":
        print(f"[preflight] {status['message']}")
        return 1
    print(f"[preflight] {status['message']}")

    system_prompt = get_full_system_prompt()
    conversation = [{"role": "user", "content": query}]
    nudged = False

    print(f"\n=== USER ===\n{query}\n")

    for round_idx in range(1, 6):
        print(f"--- LLM round {round_idx} ---")
        reply = call_llm(messages=conversation, system_prompt=system_prompt)
        reply = strip_think_tags(reply)
        print(reply)

        invocations = extract_tool_invocations(reply)
        conversation.append({"role": "assistant", "content": reply})

        if not invocations:
            announces = bool(_ANNOUNCES_TOOL.search(reply))
            refuses = bool(_REFUSES_TOOL.search(reply))
            stale_fact = bool(_REQUIRES_SEARCH.search(query)) and round_idx == 1
            if not nudged and (announces or refuses or stale_fact):
                tag = "refuses" if refuses else ("announces" if announces else "stale-fact")
                nudge = (
                    "STOP. The user's question is about current information. "
                    "Your training data is stale. Call web_search NOW with:\n"
                    "tool: web_search({\"query\": \"...\"})\n"
                    "on its own line, with nothing else. Then answer ONLY from "
                    "the snippets."
                )
                print(f"\n[nudge] {tag} → re-prompting")
                conversation.append({"role": "user", "content": nudge})
                nudged = True
                continue

            print("\n=== FINAL ASSISTANT REPLY (no more tool calls) ===")
            print(reply)
            return 0

        for name, args in invocations:
            print(f"\n>>> EXECUTING TOOL: {name}({json.dumps(args)})")
            result = execute_tool(name, args)
            preview = json.dumps(result)
            print(f"<<< RESULT ({len(preview)} chars): "
                  f"{preview[:300]}{'...' if len(preview) > 300 else ''}")
            conversation.append({
                "role": "user",
                "content": format_tool_result_for_llm(name, result),
            })

    print("[stop] hit round cap without a tool-free reply")
    return 2


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "what is the latest stable python version as of today"
    sys.exit(run(q))

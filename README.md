# LOKI

**Local Orchestrated Knowledge Interface**

A local AI assistant that actually does things. Streams its answers, searches the web when it doesn't know, and runs entirely on your machine.

No cloud. No API keys. No fluff.

---

## What is Loki?

Loki is a CLI agent harness that wraps a local LLM running in [Ollama](https://ollama.com) and gives it tools through a simple text protocol. The default brain is `qwen2.5-7B-instruct`, but the harness is model-agnostic — swap the `FROM` line in the `Modelfile` and rebuild.

Under the hood it's a ~200-line Python loop. The LLM never touches your filesystem directly: it asks for things to happen, and the harness makes them happen. Every tool call is visible in your terminal.

Built on the architecture described in [The Emperor Has No Clothes](https://www.mihaileric.com/The-Emperor-Has-No-Clothes/).

## How It Works

```
You type a message
    ↓
Loki (LLM) decides what to do
    ↓
If it needs a tool → calls it → gets result → thinks again
    ↓
If no tool needed → responds directly
    ↓
Loop continues
```

Responses **stream** to your terminal token-by-token; tool-call lines are suppressed live so you only see the answer, not the protocol.

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- The base model: `ollama pull qcwind/qwen2.5-7B-instruct-Q4_K_M`

### Setup

```bash
# Clone
git clone <your-repo-url> loki
cd loki

# Install dependencies
pip install -r requirements.txt

# Build the Loki model in Ollama (uses ./Modelfile)
ollama create loki -f Modelfile

# In a separate terminal, make sure Ollama is running
ollama serve

# Run the test suite (no Ollama needed)
python tests/test_tools.py

# Launch Loki
python loki.py
```

### What You'll See

```
             .
            / \
   |\      /   \      /|
   \ \    /     \    / /
    \ \  /       \  / /
     \ \/    _    \/ /    ╔══════════════════════════════════╗
      \__   / \   __/     ║           L O K I                ║
         |  \_/  |        ║   Calculated. Precise. AI.       ║
         \_______/        ╚══════════════════════════════════╝

Type your message and press Enter.
Type 'quit' or 'exit' to leave.
Type 'tools' to list available tools.
Type 'clear' to reset conversation history.

Checking connection to Ollama...
  ✓ Ollama is running. Model 'loki' is available.

You: what is the latest stable python version
  [ web_search ] searching the web for: "latest stable python version"
  [ web_search ] got 3 results
     1. Download Python | Python.org
        https://www.python.org/downloads/
     2. Python Latest Version - Release History, LTS & EOL
        https://versionlog.com/python/
     3. The Latest Version of Python | phoenixNAP KB
        https://phoenixnap.com/kb/latest-python-version
Loki: According to python.org, the latest stable version is Python 3.14.
```

## CLI Commands

| Command                     | What it does                          |
|-----------------------------|---------------------------------------|
| `tools`                     | List all available tools              |
| `clear` / `/clear` / `reset`| Wipe in-memory conversation history   |
| `quit` / `exit` / `q`       | Exit Loki                             |
| Ctrl+C                      | Exit Loki                             |
| Anything else               | Talk to Loki                          |

## Project Structure

```
loki/
├── loki.py              # Entry point — agent loop + StreamPrinter
├── config.py            # All constants (model, colors, ctx size)
├── system_prompt.py     # Personality + dynamic tool schema injection
├── llm.py               # Ollama API wrapper (streaming + non-streaming)
├── tool_registry.py     # Decorator-based tool registration
├── tool_parser.py       # Extracts tool calls from LLM text (forgiving)
├── spinner.py           # Stderr spinner used during time-to-first-token
├── Modelfile            # Ollama model definition
├── requirements.txt
├── tools/
│   ├── __init__.py      # Auto-discovers every .py in this folder
│   ├── read_file.py     # Read file contents
│   ├── list_files.py    # List directory contents
│   ├── edit_file.py     # Create or edit files
│   └── web_search.py    # Web search via DDGS (no API key)
└── tests/
    └── test_tools.py    # Full test suite (139 tests)
```

## Built-in Tools

| Tool          | What it does                                                  |
|---------------|---------------------------------------------------------------|
| `read_file`   | Reads a file and returns its contents                         |
| `list_files`  | Lists files and directories in a path                         |
| `edit_file`   | Creates new files or does find-and-replace edits              |
| `web_search`  | Searches the web via DDGS — no API key, aggregates DuckDuckGo, Bing, Google |

## Adding a New Tool

Drop a file in `tools/`. That's it. No other changes needed.

```python
# tools/run_command.py

from typing import Any, Dict
from tool_registry import register_tool

@register_tool("run_command")
def run_command(command: str) -> Dict[str, Any]:
    """Runs a shell command and returns the output.
    :param command: The shell command to execute.
    :return: A dict with stdout, stderr, and return code.
    """
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }
```

Restart Loki. It auto-discovers the new tool (via `pkgutil.iter_modules`), generates the schema from the function signature, and injects it into the system prompt. Loki can now run shell commands.

The schema generator reads the first line of the docstring as the description, and the type annotations + defaults from `inspect.signature` for the parameter shapes. There is no sandboxing — a tool can touch anywhere the process can.

## How Loki Uses Tools

The model emits tool calls as plain text on their own line:

```
tool: web_search({"query": "latest python version"})
```

The harness:

1. Parses the line out of the streamed response (the user never sees the raw `tool:` line — `StreamPrinter` suppresses it live).
2. Executes the registered Python function.
3. Formats the result back to the model. For `web_search`, results are reformatted from raw JSON into a numbered text block to keep small models grounded:

   ```
   Search results for "latest python version":

   [1] Download Python | Python.org
       https://www.python.org/downloads/
       Python 3.14.4 April 7, 2026 ...
   ```
4. The model produces its final answer, which streams back to your terminal.

If the model tries to answer a "current/latest/today" question from memory without searching, the harness detects it and forces a retry through `web_search`. Same goes for "I don't have a search tool" refusals — the harness corrects them and re-prompts.

## Loki's Personality

Defined in `system_prompt.py`:

- **Mandatory search** — for anything current, recent, or version-dated, Loki *must* call `web_search` before answering.
- **Grounded** — when search results are present, Loki answers only from the snippets, citing the source.
- **Opinionated** — picks a side when asked X or Y; doesn't hedge.
- **Honest** — says "I don't know" when it doesn't, prefixes guesses with `guess:`.
- **Plain** — short answers, no markdown theater, no preamble or postamble.

## Configuration

Everything tunable lives in `config.py`:

| Setting             | Default                       | What it does                          |
|---------------------|-------------------------------|---------------------------------------|
| `MODEL_NAME`        | `loki`                        | Ollama model tag the harness connects to |
| `OLLAMA_BASE_URL`   | `http://localhost:11434`      | Ollama API endpoint                   |
| `TEMPERATURE`       | `0.7`                         | Response randomness                   |
| `TOP_P`             | `0.9`                         | Nucleus sampling threshold            |
| `NUM_CTX`           | `8192`                        | Context window size                   |
| `ALLOW_THINKING`    | `False`                       | Send `think: true` to Ollama; `<think>` blocks are stripped from display either way |
| `TOOL_CALL_PREFIX`  | `tool:`                       | Line prefix the model uses to call tools |

The `Modelfile` carries the same `temperature`, `top_p`, and `num_ctx` values. Keep them in sync if you rebuild the Ollama model.

## Streaming

`llm.py` talks to Ollama's `/api/chat` with `stream: true` and parses the NDJSON line-by-line. `loki.py`'s `StreamPrinter` is a small state machine that:

- Holds the first ~5 chars of every new line and decides "tool line vs prose"
- Suppresses any line whose stripped prefix is `tool:` (case-insensitive)
- Eats `<think>...</think>` spans (rolling 9-char tail buffer)
- Emits the `Loki:` prefix lazily — a tool-call-only reply leaves no orphan label

The "thinking" spinner runs only during the time-to-first-token; the moment streaming starts, the spinner clears and tokens print live.

## Testing

```bash
# Full suite — no Ollama required, 139 tests
python tests/test_tools.py

# A single test class
python -m unittest tests.test_tools.TestStreamPrinter

# A single test
python -m unittest tests.test_tools.TestStreamPrinter.test_tool_line_is_fully_suppressed
```

A live smoke test against the running model (one full search round-trip) is at `_search_smoketest.py`.

## Architecture

Loki follows the **harness pattern**: a thin Python loop wraps the LLM and gives it tools through a text protocol.

```
User Input
    ↓
┌─────────────────────────────────┐
│  System Prompt                  │
│  (personality + tool schemas)   │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│  LLM (Ollama, streaming)        │
│  Replies with prose or          │
│  tool: NAME({"args": "..."})    │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│  StreamPrinter + Tool Parser    │
│  Live-suppress tool lines;      │
│  detect calls in accumulated    │
│  response                       │
└──────────────┬──────────────────┘
               ↓
       ┌───────┴───────┐
       │               │
   No tools         Tool call(s)
       │               │
   Done — wait     Execute via Registry
   for next            │
   prompt          Feed result back
                       │
                   Loop back to LLM
```

Every module is independent:

- Swap the LLM provider by replacing `llm.py`. Nothing else needs to change.
- Change the personality in `system_prompt.py`.
- Add tools by dropping files in `tools/`.

## Dependencies

| Package    | What for                | Required? |
|------------|-------------------------|-----------|
| `requests` | Talking to Ollama API   | Yes       |
| `ddgs`     | Web search tool         | Yes (for `web_search`) |

Everything else is Python standard library.

## License

MIT

---

*Loki streams. He picks sides. He says "I don't know" when he doesn't. And he never leaves your machine.*

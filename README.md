# LOKI

**Local Orchestrated Knowledge Interface**

A local AI assistant that actually does things. Opinionated. Honest. Runs entirely on your machine.

No cloud. No API keys. No fluff.

---

## What is Loki?

Loki is a CLI-based AI assistant powered by [Ollama](https://ollama.com) and a modular tool-calling harness. It's inspired by JARVIS — but with a personality. Loki picks sides, admits when it doesn't know something, and never fakes confidence.

Under the hood, Loki is a Python agent loop that gives a local LLM the ability to **do things** — read files, edit code, search the web — through a simple, extensible tool system. Drop a new file in `tools/` and Loki learns a new trick. No config changes, no wiring, no boilerplate.

Built on the architecture described in [The Emperor Has No Clothes](https://www.mihaileric.com/The-Emperor-Has-No-Clothes/) — a ~200-line harness pattern that turns any LLM into a capable agent.

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

The LLM never touches your filesystem directly. It **asks** for things to happen, and the harness makes them happen. Every tool call is visible in your terminal.

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- The base model: `ollama pull qwen3:8b`

### Setup

```bash
# Clone the project
git clone <your-repo-url> loki
cd loki

# Install dependencies
pip install ddgs requests

# Create the Loki model in Ollama
ollama create loki -f Modelfile

# Make sure Ollama is running
ollama serve

# Run the tests (no Ollama needed for this)
python tests/test_tools.py

# Launch Loki
python loki.py
```

### What You'll See

```
╔══════════════════════════════════════════╗
║              L O K I                     ║
║       Your Local AI Assistant            ║
╚══════════════════════════════════════════╝

Checking connection to Ollama...
  ✓ Ollama is running. Model 'loki' is available.

You: Create a hello world file
  ⠹ Pondering your question...
  ↳ edit_file({"path": "hello.py", "new_str": "print('hello')", "old_str": ""})
  ⠙ Rewriting history...
  ✓ edit_file → {"path": "/home/you/hello.py", "action": "created_file"}
  ⠸ Connecting neurons...
Loki: Done! I've created hello.py with a hello world program.
```

## CLI Commands

| Command | What it does |
|---------|-------------|
| `tools` | List all available tools |
| `quit` / `exit` / `q` | Exit Loki |
| Ctrl+C | Exit Loki |
| Anything else | Talk to Loki |

## Project Structure

```
loki/
├── loki.py              # Entry point — the agent loop
├── config.py            # All constants (model, colors, settings)
├── system_prompt.py     # Loki's personality + dynamic tool injection
├── llm.py               # Ollama API wrapper
├── tool_registry.py     # Decorator-based tool registration
├── tool_parser.py       # Extracts tool calls from LLM text
├── spinner.py           # Animated thinking/working status
├── playground.py        # Manual tool testing sandbox
├── Modelfile            # Ollama model definition
├── tools/
│   ├── __init__.py      # Auto-discovers all tools in this folder
│   ├── read_file.py     # Read file contents
│   ├── list_files.py    # List directory contents
│   ├── edit_file.py     # Create or edit files
│   └── web_search.py    # Web search via DuckDuckGo
└── tests/
    └── test_tools.py    # Full test suite (127 tests)
```

## Built-in Tools

| Tool | What it does |
|------|-------------|
| `read_file` | Reads a file and returns its contents |
| `list_files` | Lists files and directories in a path |
| `edit_file` | Creates new files or does find-and-replace edits |
| `web_search` | Searches the web via DuckDuckGo (no API key) |

## Adding a New Tool

Create a file in `tools/`. That's it. No other changes needed.

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

Restart Loki. It auto-discovers the new tool, generates the schema, and injects it into the system prompt. Loki can now run shell commands.

## Loki's Personality

Loki isn't a generic chatbot. From the Modelfile:

- **Opinionated** — picks sides and defends them with reasoning
- **Honest** — grades its own confidence (high / medium / low) and says "I don't know" without shame
- **Precise** — no filler, no fluff, every word has purpose
- **Fact-checking** — uses web search when it needs current information, never presents search results as its own knowledge
- **A thinking partner** — not a servant, not afraid to disagree

## Configuration

Everything lives in `config.py`:

| Setting | Default | What it does |
|---------|---------|-------------|
| `MODEL_NAME` | `loki` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `TEMPERATURE` | `0.7` | Response randomness |
| `TOP_P` | `0.9` | Nucleus sampling threshold |
| `NUM_CTX` | `4096` | Context window size |
| `ALLOW_THINKING` | `False` | Show qwen3's `<think>` reasoning tags |
| `LLM_TIMEOUT` | `120` | Seconds before LLM request times out |

## Testing

```bash
# Run all 127 tests (no Ollama needed)
python tests/test_tools.py

# Test tools manually in the playground
python playground.py
```

The playground lets you call tools directly, test the parser, and run a full demo workflow — all without the LLM.

## Architecture

Loki follows the **harness pattern**: a thin Python loop wraps around a local LLM, giving it tools through a text-based protocol.

```
User Input
    ↓
┌─────────────────────────────────┐
│  System Prompt                  │
│  (personality + tool schemas)   │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│  LLM (Ollama / qwen3:8b)       │
│  Responds with text or          │
│  tool: NAME({"args": "here"})   │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│  Tool Parser                    │
│  Detects tool calls in text     │
└──────────────┬──────────────────┘
               ↓
       ┌───────┴───────┐
       │               │
   No tools         Tool call(s)
       │               │
   Print response   Execute via Registry
       │               │
       │           Feed result back
       │               │
       │           Loop back to LLM
       ↓               ↓
   Wait for next user input
```

Every module is independent. Swap the LLM provider by changing `llm.py`. Change the personality in `system_prompt.py`. Add tools by dropping files. Nothing is coupled.

## Dependencies

| Package | What for | Required? |
|---------|----------|-----------|
| `requests` | Talking to Ollama API | Yes |
| `ddgs` | Web search tool | Yes (for web_search) |

Everything else is Python standard library.

## License

MIT

---

*Loki thinks before he speaks. He says "I don't know" when he doesn't. He picks sides when you ask him to. And he never leaves your machine.*

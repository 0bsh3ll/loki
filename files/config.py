"""
LOKI Configuration
All constants, model settings, and terminal styling live here.
Nothing is hardcoded elsewhere.
"""

# ─── Model Settings (from Modelfile) ───────────────────────────────
MODEL_NAME = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE = 0.7
TOP_P = 0.9
NUM_CTX = 4096

# ─── Terminal Colors ───────────────────────────────────────────────
COLOR_USER = "\033[94m"       # Blue
COLOR_ASSISTANT = "\033[93m"  # Yellow
COLOR_TOOL = "\033[92m"       # Green
COLOR_THINKING = "\033[90m"   # Gray
COLOR_ERROR = "\033[91m"      # Red
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"

# ─── Tool Call Format ──────────────────────────────────────────────
# The prefix the LLM uses to signal a tool invocation.
# Example: tool: read_file({"filename": "main.py"})
TOOL_CALL_PREFIX = "tool:"

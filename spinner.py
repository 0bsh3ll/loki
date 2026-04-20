"""
LOKI Spinner

Shows animated status messages in the terminal while the LLM
is thinking or a tool is executing. Runs in a background thread
so the main thread can block on I/O without freezing the UI.

Usage:
    with Spinner("thinking"):
        response = call_llm(...)

    with Spinner("tool", tool_name="web_search"):
        result = execute_tool(...)
"""

import itertools
import random
import sys
import threading
import time

from config import COLOR_THINKING, COLOR_TOOL, COLOR_RESET


# ─── Spinner Frames ────────────────────────────────────────────────
# Braille dot animation — smooth and compact
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# ─── Status Messages ───────────────────────────────────────────────
# Random messages shown while waiting. Keeps things lively.

THINKING_MESSAGES = [
    "Turning gears...",
    "Pondering your question...",
    "Consulting the runes...",
    "Connecting neurons...",
    "Weighing the options...",
    "Thinking it through...",
    "Processing...",
    "Brewing an answer...",
    "Running the numbers...",
    "Considering the angles...",
    "Assembling thoughts...",
    "Crunching the logic...",
]

TOOL_MESSAGES = {
    "read_file": [
        "Reading the sacred scrolls...",
        "Scanning file contents...",
        "Opening the file...",
        "Parsing the document...",
    ],
    "list_files": [
        "Rummaging through folders...",
        "Surveying the directory...",
        "Cataloguing files...",
        "Mapping the filesystem...",
    ],
    "edit_file": [
        "Rewriting history...",
        "Tweaking the code...",
        "Making edits...",
        "Applying changes...",
    ],
    "web_search": [
        "Scouring the internet...",
        "Asking the oracle...",
        "Searching the web...",
        "Hunting for answers...",
        "Querying the void...",
    ],
    "_default": [
        "Running the tool...",
        "Executing...",
        "Working on it...",
        "Processing the request...",
    ],
}


def _get_message(mode: str, tool_name: str = "") -> str:
    """Pick a random status message based on mode and tool name."""
    if mode == "thinking":
        return random.choice(THINKING_MESSAGES)
    elif mode == "tool":
        messages = TOOL_MESSAGES.get(tool_name, TOOL_MESSAGES["_default"])
        return messages[random.randint(0, len(messages) - 1)]
    else:
        return "Working..."


class Spinner:
    """
    Animated spinner with status messages.

    Can be used as a context manager:
        with Spinner("thinking"):
            slow_operation()

    Or manually:
        s = Spinner("tool", tool_name="web_search")
        s.start()
        ...
        s.stop()
    """

    def __init__(self, mode: str = "thinking", tool_name: str = ""):
        """
        Args:
            mode: "thinking" for LLM wait, "tool" for tool execution.
            tool_name: Name of the tool being executed (used for
                       tool-specific messages). Ignored if mode is "thinking".
        """
        self.mode = mode
        self.tool_name = tool_name
        self._running = False
        self._thread = None
        self._message = _get_message(mode, tool_name)

    def _spin(self):
        """Background thread loop — animates the spinner."""
        color = COLOR_THINKING if self.mode == "thinking" else COLOR_TOOL
        frames = itertools.cycle(SPINNER_FRAMES)
        message_timer = 0

        while self._running:
            frame = next(frames)
            line = f"\r{color}{frame} {self._message}{COLOR_RESET}"
            sys.stderr.write(line)
            sys.stderr.flush()
            time.sleep(0.08)

            # Rotate the message every ~3 seconds
            message_timer += 0.08
            if message_timer >= 3.0:
                self._message = _get_message(self.mode, self.tool_name)
                message_timer = 0

        # Clear the spinner line when done
        sys.stderr.write("\r" + " " * 60 + "\r")
        sys.stderr.flush()

    def start(self):
        """Start the spinner animation in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the spinner and clean up the terminal line."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

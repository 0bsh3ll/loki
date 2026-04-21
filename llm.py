"""
LOKI LLM Client

Thin wrapper around Ollama's REST API. The agent loop calls
`call_llm()` and gets back a string. Everything about the
model connection is isolated here — swap providers by changing
only this file.

Requires: Ollama running locally with the 'loki' model created.
  ollama create loki -f Modelfile
"""

import json
from typing import Any, Dict, List, Optional

import requests

from config import MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE, TOP_P, NUM_CTX,ALLOW_THINKING


def call_llm(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    stream: bool = False,
) -> str:
    """
    Sends a conversation to the Ollama API and returns the
    assistant's response as a string.

    Args:
        messages: Conversation history. Each dict has 'role' and 'content'.
                  Roles: 'user', 'assistant', 'system'
        system_prompt: Optional system prompt. If provided, it's sent
                       separately via Ollama's 'system' field (merges
                       with the Modelfile's built-in prompt).
        model: Model name override. Defaults to config.MODEL_NAME.
        stream: If True, returns chunks as they arrive. (Not yet used,
                reserved for Phase 5 spinner integration.)

    Returns:
        The assistant's response text.

    Raises:
        ConnectionError: If Ollama is not running or unreachable.
        RuntimeError: If the API returns an error response.
    """
    model = model or MODEL_NAME
    url = f"{OLLAMA_BASE_URL}/api/chat"

    # ── Build the request payload ──────────────────────────────
    # Separate system messages from the conversation — Ollama
    # takes the system prompt as a top-level field.
    filtered_messages = []
    for msg in messages:
        if msg["role"] == "system":
            # If system_prompt wasn't explicitly passed, use this
            if system_prompt is None:
                system_prompt = msg["content"]
        else:
            filtered_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    payload: Dict[str, Any] = {
        "model": model,
        "messages": filtered_messages,
        "stream": False,
        "think" : ALLOW_THINKING,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_ctx": NUM_CTX,
        }
    }

    if system_prompt:
        payload["system"] = system_prompt

    # ── Make the request ───────────────────────────────────────
    try:
        response = requests.post(url, json=payload, timeout=1000)
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Is Ollama running? Start it with: ollama serve"
        )
    except requests.Timeout:
        raise RuntimeError(
            f"Ollama request timed out after 120s. "
            f"The model might be loading or the query too complex."
        )

    # ── Parse the response ─────────────────────────────────────
    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama API error {response.status_code}: {response.text}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Ollama returned invalid JSON: {response.text[:200]}"
        )

    # Ollama chat response structure:
    # {"message": {"role": "assistant", "content": "..."}, ...}
    message = data.get("message", {})
    content = message.get("content", "")

    if not content:
        raise RuntimeError(
            f"Ollama returned empty response. Full payload: {json.dumps(data)[:300]}"
        )

    return content


def check_ollama_connection() -> Dict[str, Any]:
    """
    Quick health check — is Ollama running and is the model available?

    Returns:
        A dict with 'status' ('ok' or 'error') and details.
    """
    # Check if Ollama is running
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    except requests.ConnectionError:
        return {
            "status": "error",
            "message": f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Run: ollama serve"
        }
    except requests.Timeout:
        return {
            "status": "error",
            "message": f"Ollama at {OLLAMA_BASE_URL} is not responding."
        }

    if resp.status_code != 200:
        return {
            "status": "error",
            "message": f"Ollama API error: {resp.status_code}"
        }

    # Check if our model exists
    try:
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]

        # Ollama model names can be "loki:latest" or just "loki"
        model_found = any(
            m == MODEL_NAME or m.startswith(f"{MODEL_NAME}:")
            for m in models
        )

        if model_found:
            return {
                "status": "ok",
                "message": f"Ollama is running. Model '{MODEL_NAME}' is available.",
                "models": models
            }
        else:
            return {
                "status": "error",
                "message": (
                    f"Ollama is running but model '{MODEL_NAME}' not found. "
                    f"Create it with: ollama create {MODEL_NAME} -f Modelfile\n"
                    f"Available models: {', '.join(models) if models else '(none)'}"
                ),
                "models": models
            }

    except (json.JSONDecodeError, KeyError) as e:
        return {
            "status": "error",
            "message": f"Unexpected Ollama response: {e}"
        }

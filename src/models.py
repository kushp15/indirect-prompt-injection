"""
models.py — Standalone model definitions for the baseline evaluation.

Only OllamaModel is implemented here; it talks to a locally-running
Ollama server through the OpenAI-compatible /v1 endpoint.
"""

import os
import time

# Load .env if available (optional convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class BaseModel:
    """Minimal interface every model must implement."""

    def prepare_input(self, sys_prompt: str, user_prompt: str):
        raise NotImplementedError

    def call_model(self, model_input) -> str:
        raise NotImplementedError


class OllamaModel(BaseModel):
    """
    Wraps a locally-running Ollama instance exposed on the OpenAI-compatible
    /v1 endpoint (default: http://localhost:11434/v1).

    Set OLLAMA_BASE_URL in the environment (or .env) to override the host.
    """

    def __init__(self, params: dict):
        from openai import OpenAI

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.model_name: str = params["model_name"]

    def prepare_input(self, sys_prompt: str, user_prompt: str) -> list:
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt},
        ]

    def call_model(self, model_input: list, max_retries: int = 3, retry_delay: int = 2) -> str:
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    messages=model_input,
                    model=self.model_name,
                    temperature=0,
                    timeout=120,
                )
                return completion.choices[0].message.content
            except Exception as exc:
                msg = str(exc)
                if "connection" in msg.lower():
                    print(
                        "Ollama connection error — "
                        "make sure 'ollama serve' is running on localhost:11434"
                    )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to get a response from Ollama after {max_retries} attempts.")
                    raise


# Registry — extend here when adding new model types
MODELS = {
    "Ollama": OllamaModel,
}

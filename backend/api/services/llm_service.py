"""Generates the final natural-language explanation by sending a grounded
prompt (built by routers/explain.py from real prediction + RAG context) to
Groq's OpenAI-compatible chat completions API.

Deliberately has no REQUIRED_FILES / warm_up() and is never added to
api/main.py's fail-fast startup checks: unlike the model and the knowledge
base, whether this works depends on an external API key that may not be
set in every environment, and /explain must degrade to a clear message in
that case instead of taking the whole app down or 500-ing.
"""

import os

from openai import OpenAI

from ..logging_config import logger

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

NOT_CONFIGURED_MESSAGE = (
    "AI explanation unavailable: GROQ_API_KEY is not set on the server. "
    "The risk prediction above is still fully valid -- only the natural-"
    "language explanation requires an LLM API key."
)


def is_configured() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def _create_client(api_key: str) -> OpenAI:
    """Isolated on purpose: tests monkeypatch this one function to stub out
    the network call, instead of needing a real Groq API key to exercise
    the "key is configured" code path.
    """
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def generate_explanation(system_prompt: str, user_prompt: str) -> dict:
    """Returns {"explanation": str, "grounded": bool}. Never raises --
    a missing key or a failed API call both produce a clear, readable
    explanation string with grounded=False rather than an exception, so
    callers (routers/explain.py) can't accidentally turn this into a 500.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"explanation": NOT_CONFIGURED_MESSAGE, "grounded": False}

    try:
        client = _create_client(api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        text = response.choices[0].message.content.strip()
        return {"explanation": text, "grounded": True}
    except Exception as exc:
        logger.exception("Groq explanation request failed")
        return {
            "explanation": f"AI explanation unavailable: the Groq API request failed ({exc}).",
            "grounded": False,
        }

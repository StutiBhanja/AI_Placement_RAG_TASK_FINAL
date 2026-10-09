"""Gemini LLM + embeddings, with safe error handling.

Why this file exists
--------------------
The Gemini free tier is small. Typical errors:
  * 429 RESOURCE_EXHAUSTED -> quota finished (per model, per minute or per day)
  * 503 UNAVAILABLE        -> model is busy for a moment
Instead of crashing, `safe_invoke` retries short problems and raises one clear
exception (`LLMUnavailable`) that the pipelines and the Streamlit app can show nicely.
"""
from __future__ import annotations

import re
import time
import warnings

from . import config

warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")


class LLMUnavailable(Exception):
    """Gemini could not be used. `kind` is 'quota', 'busy', 'auth' or 'other'."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


def classify_error(err: Exception) -> str:
    text = str(err)
    low = text.lower()
    if "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in low:
        return "quota"
    if "503" in text or "UNAVAILABLE" in text or "overloaded" in low or "high demand" in low:
        return "busy"
    if ("API key" in text or "API_KEY" in text or "401" in text or "403" in text
            or "PERMISSION_DENIED" in text or "UNAUTHENTICATED" in text):
        return "auth"
    return "other"


def retry_delay_seconds(err: Exception) -> float | None:
    """Read Gemini's suggested wait (seconds) from the error text, if present."""
    text = str(err)
    m = (re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", text)
         or re.search(r"retry in (\d+(?:\.\d+)?)s", text))
    return float(m.group(1)) if m else None


FRIENDLY = {
    "quota": ("Gemini quota/rate limit reached. Wait for the quota to reset, add another model "
              "name to LLM_MODELS in .env, or use an API key from a project with available quota."),
    "busy": "Gemini is temporarily busy (503). Please try again in a few seconds.",
    "auth": "Gemini rejected the API key. Check GOOGLE_API_KEY in your .env file.",
    "other": "Gemini returned an unexpected error.",
}


def safe_invoke(chain, inputs: dict, *, max_wait: float = 20.0, busy_retries: int = 2):
    """Run `chain.invoke(inputs)`; retry short 503/429 problems; raise LLMUnavailable otherwise."""
    attempt = 0
    while True:
        try:
            return chain.invoke(inputs)
        except Exception as err:  # noqa: BLE001 - we classify every provider error
            kind = classify_error(err)
            attempt += 1
            if kind == "busy" and attempt <= busy_retries:
                time.sleep(2 * attempt)
                continue
            if kind == "quota":
                wait = retry_delay_seconds(err)
                if attempt == 1 and wait is not None and wait <= max_wait:
                    time.sleep(wait + 1)            # per-minute limit: one short wait is enough
                    continue
            detail = str(err)[:300].replace("\n", " ")
            raise LLMUnavailable(kind, f"{FRIENDLY[kind]} [{type(err).__name__}: {detail}]") from err


def make_embeddings(model: str | None = None):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    config.get_api_key()
    return GoogleGenerativeAIEmbeddings(model=model or config.EMBEDDING_MODEL)


def make_llm(models: list[str] | None = None, temperature: float = 0.0):
    """Gemini chat model. With several model names, the next one is tried when one fails."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    config.get_api_key()
    names = models or config.LLM_MODELS
    llms = [
        ChatGoogleGenerativeAI(model=n, temperature=temperature, max_retries=1, timeout=60)
        for n in names
    ]
    return llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]

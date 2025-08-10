from __future__ import annotations

import os
from typing import Dict, Any

# Streamlit is optional here; the module should not crash if it's unavailable.
try:  # pragma: no cover
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore


def _env_or_secret(name: str) -> str:
    """Prefer environment variable; fall back to Streamlit secrets if available.
    Never raise if secrets.toml is missing."""
    val = os.getenv(name, "")
    if val:
        return val
    if st is None:
        return ""
    try:
        # Access inside try so missing secrets.toml doesn't crash
        return str(st.secrets[name])
    except Exception:
        return ""


def _get_keys() -> Dict[str, str]:
    return {
        "openai": _env_or_secret("OPENAI_API_KEY"),
        "google": _env_or_secret("GOOGLE_API_KEY"),
    }


def get_base_prompt(features: Dict[str, int]) -> str:
    """Compose a concise incident context for the LLM."""
    lines = [
        "You are a SOC analyst. Produce a short, actionable response plan.",
        "Inputs are URL features; values are -1/0/1 booleans/tri-states.",
        "Return a compact plan with: risk_level, summary, recommended_actions, "
        "and a brief communication_draft for stakeholders.",
        "",
        "Features:",
    ]
    for k, v in features.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _local_fallback() -> Dict[str, Any]:
    return {
        "risk_level": "High",
        "summary": "Suspicious URL consistent with phishing; containment and triage required.",
        "recommended_actions": [
            "Block the URL at the gateway and mail filters.",
            "Quarantine any emails containing the URL.",
            "Search SIEM for requests to this URL and related domains (last 7 days).",
            "Notify potentially affected users and reset credentials as needed.",
            "Open an incident ticket and attach indicators/evidence.",
        ],
        "communication_draft": (
            "We detected and blocked a suspicious URL while we investigate. "
            "Please avoid clicking similar links. We will share next steps if action is required."
        ),
    }


def _with_openai(api_key: str, prompt: str) -> Dict[str, Any]:
    # Import lazily so the module loads even if package isn't installed.
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)
    # Ask the model to return strict JSON.
    sys_msg = (
        "Return a STRICT JSON object with keys: risk_level, summary, "
        "recommended_actions (list of short strings), communication_draft."
    )
    msg = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]
    # Use a small, fast model; adjust if you prefer.
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msg,
        temperature=0.2,
    )
    text = resp.choices[0].message.content or ""
    # Best effort JSON parse; fall back to a simple wrapper if parsing fails.
    try:
        import json

        data = json.loads(text)
        # minimal shape check
        if isinstance(data, dict) and "summary" in data:
            return data
    except Exception:
        pass
    return {
        "risk_level": "Medium",
        "summary": text.strip()[:800],
        "recommended_actions": _local_fallback()["recommended_actions"],
        "communication_draft": _local_fallback()["communication_draft"],
    }


def _with_gemini(api_key: str, prompt: str) -> Dict[str, Any]:
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    sys = (
        "Return a STRICT JSON object with keys: risk_level, summary, "
        "recommended_actions (list), communication_draft."
    )
    resp = model.generate_content([sys, prompt])
    text = resp.text or ""
    try:
        import json

        data = json.loads(text)
        if isinstance(data, dict) and "summary" in data:
            return data
    except Exception:
        pass
    return {
        "risk_level": "Medium",
        "summary": text.strip()[:800],
        "recommended_actions": _local_fallback()["recommended_actions"],
        "communication_draft": _local_fallback()["communication_draft"],
    }


def generate_prescription(provider: str, features: Dict[str, int]) -> Dict[str, Any]:
    """Main entry point used by the Streamlit app."""
    prompt = get_base_prompt(features)
    keys = _get_keys()
    name = (provider or "").lower().strip()

    try:
        if name.startswith("openai") and keys["openai"]:
            return _with_openai(keys["openai"], prompt)
        if name.startswith("gemini") and keys["google"]:
            return _with_gemini(keys["google"], prompt)
    except Exception:
        # Any provider error → safe fallback
        return _local_fallback()

    # No keys or unknown provider → fallback
    return _local_fallback()

"""
Build a prescriptive incident-response plan from alert features.

- Reads API keys from Streamlit secrets or environment variables.
- Supports OpenAI or Gemini; falls back to a local plan if anything fails.
"""

import json
import os
from typing import Any, Dict

import streamlit as st
import google.generativeai as genai

try:
    from openai import OpenAI  # modern SDK
    _OPENAI_NEW = True
except Exception:  # pragma: no cover
    import openai  # legacy
    _OPENAI_NEW = False


def _get_keys() -> Dict[str, str]:
    openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    return {"openai": openai_key, "gemini": gemini_key}


def _base_prompt(alert: Dict[str, Any]) -> str:
    return (
        "You are an expert SOAR system.\n"
        "Return ONLY a JSON object with keys exactly:\n"
        '  "summary" (string),\n'
        '  "risk_level" (string: "Critical","High","Medium","Low"),\n'
        '  "recommended_actions" (array of strings),\n'
        '  "communication_draft" (string).\n\n'
        "Use concise, actionable language.\n\n"
        f"Alert details:\n{json.dumps(alert, indent=2)}\n"
    )


def _coerce_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json", "", 1).strip()
    if cleaned.startswith("'''"):
        cleaned = cleaned.strip("'")
    return json.loads(cleaned)


def _gemini(alert: Dict[str, Any], key: str) -> Dict[str, Any]:
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(_base_prompt(alert))
    return _coerce_json(resp.text or "")


def _openai(alert: Dict[str, Any], key: str) -> Dict[str, Any]:
    prompt = _base_prompt(alert)
    if _OPENAI_NEW:
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    # legacy SDK path
    openai.api_key = key  # type: ignore[attr-defined]
    resp = openai.ChatCompletion.create(  # type: ignore[attr-defined]
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return _coerce_json(resp["choices"][0]["message"]["content"])


def _fallback(alert: Dict[str, Any]) -> Dict[str, Any]:
    risky = (
        alert.get("Abnormal_URL") == 1
        or alert.get("Shortining_Service") == 1
        or alert.get("having_IP_Address") == 1
    )
    risk = "High" if risky else "Medium"
    return {
        "summary": (
            "Suspicious URL consistent with phishing activity; containment and triage required."
        ),
        "risk_level": risk,
        "recommended_actions": [
            "Block the URL at the gateway and mail filters.",
            "Quarantine any emails containing the URL.",
            "Search SIEM for requests to this URL and related domains (last 7 days).",
            "Notify potentially affected users and reset credentials as needed.",
            "Open an incident ticket and attach indicators/evidence.",
        ],
        "communication_draft": (
            "We detected a suspicious URL indicative of phishing. Access has been blocked "
            "while we investigate potential exposure. If you clicked the link or entered "
            "credentials, please report it and reset your password immediately."
        ),
    }


def generate_prescription(provider: str, alert: Dict[str, Any]) -> Dict[str, Any]:
    """Unified entry point for the Streamlit app."""
    keys = _get_keys()
    p = (provider or "").strip().lower()
    try:
        if p.startswith("openai") and keys["openai"]:
            return _openai(alert, keys["openai"])
        if p.startswith("gemini") and keys["gemini"]:
            return _gemini(alert, keys["gemini"])
        return _fallback(alert)
    except Exception:
        return _fallback(alert)

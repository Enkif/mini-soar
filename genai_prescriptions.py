"""
genai_prescriptions.py
Builds a prescriptive incident-response plan from alert features.

- Reads API keys from Streamlit secrets or environment variables.
- Works with OpenAI or Gemini, and gracefully falls back to a local plan if
  keys are missing or provider calls fail.
"""

import os
import json
import streamlit as st
import google.generativeai as genai

# OpenAI: prefer new SDK; fall back to legacy if user has an older version
try:
    from openai import OpenAI  # new SDK style
    _OPENAI_NEW = True
except Exception:  # pragma: no cover
    import openai  # legacy
    _OPENAI_NEW = False

# ---- Keys: prefer Streamlit secrets; fall back to environment variables ----
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

OPENAI_OK = bool(OPENAI_KEY)
GEMINI_OK = False
if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        GEMINI_OK = True
    except Exception:
        GEMINI_OK = False


def get_base_prompt(alert_details: dict) -> str:
    """Return a strict prompt asking the model for JSON only."""
    return (
        "You are an expert SOAR system.\n"
        "Return ONLY a JSON object with keys exactly:\n"
        '  "summary" (string),\n'
        '  "risk_level" (string: e.g., "Critical","High","Medium","Low"),\n'
        '  "recommended_actions" (array of strings),\n'
        '  "communication_draft" (string).\n\n'
        "Use concise, actionable language.\n\n"
        f"Alert details:\n{json.dumps(alert_details, indent=2)}\n"
    )


def _coerce_json(text: str) -> dict:
    """
    Try to parse JSON from model output and tolerate code fences.
    Raise on failure so caller can trigger fallback.
    """
    cleaned = (text or "").strip()
    # remove common code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    if cleaned.startswith("'''"):
        cleaned = cleaned.strip("'")
    return json.loads(cleaned)


# ---------------- Providers ----------------
def get_gemini_prescription(alert_details: dict) -> dict:
    if not GEMINI_OK:
        raise RuntimeError("Gemini key not set")
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = get_base_prompt(alert_details)
    resp = model.generate_content(prompt)
    return _coerce_json(resp.text or "")


def get_openai_prescription(alert_details: dict) -> dict:
    if not OPENAI_OK:
        raise RuntimeError("OpenAI key not set")
    prompt = get_base_prompt(alert_details)

    if _OPENAI_NEW:
        client = OpenAI(api_key=OPENAI_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    else:
        # Legacy SDK
        openai.api_key = OPENAI_KEY
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        return _coerce_json(resp["choices"][0]["message"]["content"])


# ---------------- Local fallback ----------------
def _local_fallback(alert_details: dict) -> dict:
    # Extremely simple heuristic—works even with no keys.
    risky = (
        (alert_details.get("Abnormal_URL") == 1)
        or (alert_details.get("Shortining_Service") == 1)
        or (alert_details.get("having_IP_Address") == 1)
    )
    risk = "High" if risky else "Medium"
    return {
        "summary": "Suspicious URL consistent with phishing activity; containment and triage required.",
        "risk_level": risk,
        "recommended_actions": [
            "Block the URL at the secure web gateway and mail filters.",
            "Quarantine any emails containing this URL.",
            "Search SIEM for clicks/requests to this URL and related domains (last 7 days).",
            "Notify potentially affected users with credential-reset guidance.",
            "Open an incident ticket and attach indicators/evidence.",
        ],
        "communication_draft": (
            "We detected a suspicious URL indicative of phishing. Access has been blocked while "
            "we investigate potential exposure. If you clicked the link or entered credentials, "
            "please report it and reset your password immediately."
        ),
    }


# ---------------- Public entry point ----------------
def generate_prescription(provider: str, alert_details: dict) -> dict:
    """
    Unified entry point for the Streamlit app.
    provider: 'OpenAI', 'Gemini', or anything else → fallback
    """
    p = (provider or "").strip().lower()
    try:
        if p.startswith("openai"):
            return get_openai_prescription(alert_details)
        if p.startswith("gemini"):
            return get_gemini_prescription(alert_details)
        return _local_fallback(alert_details)
    except Exception:
        # Any provider error → safe local plan so the app never crashes
        return _local_fallback(alert_details)


__all__ = ["generate_prescription"]

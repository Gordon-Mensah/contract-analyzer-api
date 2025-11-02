# core/utils.py

import html, re, json
from hashlib import sha256
from deep_translator import GoogleTranslator

RISK_HIGHLIGHTS = {
    "indemnify": "background:#ffd6d6",
    "exclusive": "background:#ffd6d6",
    "penalty": "background:#ffd6d6",
    "binding": "background:#fff0b3",
    "termination": "background:#fff0b3"
}

def mkhash(*args) -> str:
    h = sha256()
    for a in args:
        if isinstance(a, str):
            h.update(a.encode("utf-8"))
        else:
            h.update(json.dumps(a, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()

def translate_to_hungarian(text):
    try:
        return GoogleTranslator(source='auto', target='hu').translate(text)
    except Exception:
        return ""

def highlight_risks(text):
    safe = html.escape(text)
    for term, style in RISK_HIGHLIGHTS.items():
        safe = re.sub(fr"(?i)\b({re.escape(term)})\b", rf"<span style='{style};padding:2px;border-radius:3px'>\1</span>", safe)
    return safe

def format_badges(clause_type, risk_level):
    risk_colors = {
        "High": "🔴 High Risk",
        "Medium": "🟠 Medium Risk",
        "Low": "🟢 Low Risk"
    }
    return f"📌 **{clause_type}** | {risk_colors.get(risk_level, '⚪ Unknown Risk')}"

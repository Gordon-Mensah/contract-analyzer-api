import os
import re
from docx import Document
import pdfplumber
from core.clause_explanations import clause_type_explanations

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN_SPLITTER = True
except Exception:
    _HAS_LANGCHAIN_SPLITTER = False

from transformers import pipeline
from functools import lru_cache

@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ---------- Clause Detection Maps ----------
keyword_map = {
    "nda": {
        "Confidentiality": [
            "confidential", "non-disclosure", "secret", "proprietary", "private", "classified", "sensitive", "internal"
        ],
        "Restrictions": [
            "reverse engineer", "copy", "replicate", "duplicate", "unauthorized", "prohibit", "ban", "limit", "restrict"
        ],
        "Termination": [
            "terminate", "end", "cancel", "conclude", "expire", "cease", "withdraw", "revoke"
        ]
    },
    "rental": {
        "Payment": [
            "rent", "deposit", "fee", "dues", "installment", "charge", "billing", "cost"
        ],
        "Termination": [
            "eviction", "terminate", "notice", "vacate", "end lease", "cancel", "quit", "release"
        ],
        "Maintenance": [
            "repair", "damage", "cleaning", "upkeep", "fix", "restore", "service", "maintain"
        ],
        "Liability": [
            "insurance", "liability", "damages", "responsibility", "accountable", "fault", "risk", "cover"
        ]
    },
    "employment": {
        "Duties": [
            "responsibilities", "tasks", "role", "reporting", "obligations", "functions", "assignments"
        ],
        "Compensation": [
            "salary", "bonus", "benefits", "pay", "wages", "income", "remuneration", "package"
        ],
        "Termination": [
            "resignation", "dismissal", "notice", "severance", "layoff", "exit", "release"
        ],
        "IP": [
            "intellectual property", "invention", "ownership", "patent", "copyright", "trademark", "creation"
        ]
    },
    "service": {
        "Scope": [
            "services", "deliverables", "timeline", "schedule", "coverage", "extent", "range", "tasks"
        ],
        "Payment": [
            "fee", "invoice", "payment terms", "cost", "charge", "rate", "billing"
        ],
        "Termination": [
            "cancel", "terminate", "breach", "end", "revoke", "discontinue", "cease"
        ],
        "Liability": [
            "indemnify", "damages", "limitation", "responsibility", "risk", "cover", "accountability"
        ]
    },
    "sales": {
        "Price": [
            "price", "cost", "payment", "rate", "charge", "amount", "value"
        ],
        "Delivery": [
            "shipment", "delivery", "timeline", "dispatch", "send", "transport", "arrival"
        ],
        "Warranty": [
            "warranty", "guarantee", "defect", "assurance", "coverage", "promise", "quality"
        ],
        "Returns": [
            "refund", "return", "exchange", "credit", "replacement", "cancel", "reverse"
        ]
    },
    "other": {
        "General": [
            "agreement", "party", "terms", "conditions", "contract", "deal", "understanding"
        ]
    }
}

risk_terms = {
    "High": ["penalty", "exclusive", "binding", "indemnify", "irreversible"],
    "Medium": ["termination", "confidential", "governing law", "non-compete"],
    "Low": ["notice", "duration", "payment", "invoice"]
}

# ---------- Core Functions ----------
def load_contract(file):
    if file.name.lower().endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file.name.lower().endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return file.read().decode("utf-8", errors="ignore")

def _simple_chunker(text, chunk_size=500, chunk_overlap=50):
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        if end >= n:
            chunks.append(text[start:n].strip())
            break
        split_at = None
        window = text[start:end]
        for sep in ("\n\n", "\n", ". ", "; ", ", "):
            pos = window.rfind(sep)
            if pos != -1 and pos > int(chunk_size * 0.4):
                split_at = start + pos + len(sep)
                break
        if split_at is None:
            split_at = end
        chunk = text[start:split_at].strip()
        chunks.append(chunk)
        start = split_at - chunk_overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]

def chunk_contract(text, chunk_size=500, chunk_overlap=50):
    if _HAS_LANGCHAIN_SPLITTER:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)
    else:
        return _simple_chunker(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def normalize(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower())

def label_clause(text, contract_type="generic"):
    text_l = normalize(text)
    clause_type = "Other"
    risk_level = "Low"

    if contract_type in keyword_map:
        for label, terms in keyword_map[contract_type].items():
            for term in terms:
                term_norm = normalize(term)
                if term_norm in text_l:
                    clause_type = label
                    break
            if clause_type != "Other":
                break

    for level, terms in risk_terms.items():
        for term in terms:
            term_norm = normalize(term)
            if term_norm in text_l:
                risk_level = level
                break
        if risk_level != "Low":
            break

    return clause_type, risk_level

def explain_clause_risk(clause_text, clause_type, risk_level):
    if risk_level == "High":
        return "⚠️ This clause may expose you to significant legal or financial risk. Consider negotiating safer terms."
    elif risk_level == "Medium":
        return "⚠️ This clause has moderate risk. Review it carefully and consider if it aligns with your needs."
    elif risk_level == "Low":
        return "✅ This clause is generally safe and common in contracts."
    return ""

def get_clause_explanation(clause_type):
    return clause_type_explanations.get(clause_type, "This clause type is not yet explained.")

def explain_clause_text(text):
    text_lower = text.lower()
    if "termination" in text_lower:
        return "This clause explains how and when the contract can be ended by either party."
    elif "confidential" in text_lower or "nda" in text_lower:
        return "This clause ensures that sensitive information shared between parties remains private."
    elif "payment" in text_lower:
        return "This clause outlines how and when payments will be made."
    elif "liability" in text_lower:
        return "This clause defines who is responsible if something goes wrong."
    else:
        return "This clause covers general terms and conditions related to the agreement."

def summarize_contract(text):
    summarizer = get_summarizer()
    short_text = text[:2000]  # Limit to first 2000 characters
    try:
        out = summarizer(short_text, max_length=300, min_length=100, do_sample=False)
        return out[0]["summary_text"] if isinstance(out, list) else out.get("summary_text", "")
    except Exception as e:
        return f"⚠️ Unable to summarize contract. Error: {str(e)}"




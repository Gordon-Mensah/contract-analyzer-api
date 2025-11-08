import os
import re
from docx import Document
import pdfplumber
import warnings
from core.clause_explanations import clause_type_explanations

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN_SPLITTER = True
except Exception:
    _HAS_LANGCHAIN_SPLITTER = False



# ---------- Clause Detection Maps ----------
keyword_map = {
    "nda": {
            "Confidentiality": [
                "confidential", "non-disclosure", "secret", "proprietary", "private", "classified", "sensitive", "internal",
                "nonpublic", "restricted", "undisclosed", "privacy", "secure", "protected information",
                "data protection", "confidential materials", "security protocol", "access restriction", "information control",
                "trade secret", "non-circulation", "non-public", "restricted access", "data classification"
        ],
            "Restrictions": [
                "reverse engineer", "copy", "replicate", "duplicate", "unauthorized", "prohibit", "ban", "limit", "restrict",
                "disclose", "use limitation", "access control", "redistribute", "circumvent",
                "derive", "reproduce", "transmit", "extract", "misuse", "interfere", "tamper",
                "non-use", "non-sharing", "non-transfer", "non-distribution", "non-reproduction"
        ],
        "Termination": [
                "terminate", "end", "cancel", "conclude", "expire", "cease", "withdraw", "revoke",
                "termination date", "duration", "survival", "post-termination",
                "expiration", "termination clause", "validity period", "termination notice",
                "sunset clause", "contract end", "termination trigger"
        ]

    },
    "rental": {
        "Payment": [
            "rent", "deposit", "fee", "dues", "installment", "charge", "billing", "cost",
            "late fee", "monthly", "security deposit", "utilities", "arrears",
            "rental amount", "payment schedule", "due date", "nonpayment", "financial obligation",
            "rent increase", "rent adjustment", "payment default"
    ],
        "Termination": [
            "eviction", "terminate", "notice", "vacate", "end lease", "cancel", "quit", "release",
            "early termination", "break lease", "non-renewal", "move-out",
            "termination clause", "lease expiration", "termination rights", "termination conditions",
            "lease breach", "termination penalty", "termination fee"
        ],
        "Maintenance": [
            "repair", "damage", "cleaning", "upkeep", "fix", "restore", "service", "maintain",
            "wear and tear", "maintenance request", "inspection", "condition", "replace",
            "maintenance responsibility", "property condition", "repairs required", "cleanliness",
            "tenant obligations", "landlord duties", "repair timeline"
        ],
        "Liability": [
            "insurance", "liability", "damages", "responsibility", "accountable", "fault", "risk", "cover",
            "negligence", "indemnify", "loss", "hazard", "incident",
            "liability waiver", "tenant responsibility", "property damage", "third-party claims",
            "accident", "injury", "legal exposure"
        ]

    },
    "employment": {
        "Duties": [
            "responsibilities", "tasks", "role", "reporting", "obligations", "functions", "assignments",
            "job description", "performance", "expectations", "scope of work",
            "duties", "workload", "job title", "chain of command", "supervision",
            "job responsibilities", "work expectations", "employee obligations"
        ],
        "Compensation": [
            "salary", "bonus", "benefits", "pay", "wages", "income", "remuneration", "package",
            "equity", "stock options", "commission", "pension", "reimbursement",
            "compensation structure", "pay frequency", "incentives", "financial package",
            "variable pay", "performance bonus", "compensation review"
        ],
        "Termination": [
            "resignation", "dismissal", "notice", "severance", "layoff", "exit", "release",
            "termination clause", "cause", "at-will", "final paycheck", "termination date",
            "termination rights", "termination process", "exit interview", "termination conditions",
            "termination benefits", "termination notice", "termination agreement"
        ],
        "IP": [
            "intellectual property", "invention", "ownership", "patent", "copyright", "trademark", "creation",
            "work product", "assign", "developed during employment", "moral rights",
            "IP rights", "innovation", "proprietary work", "employee inventions",
            "IP assignment", "ownership clause", "creative output"
        ]

    },
    "service": {
        "Scope": [
            "services", "deliverables", "timeline", "schedule", "coverage", "extent", "range", "tasks",
            "milestones", "project plan", "statement of work", "SOW", "service levels"
        ],
        "Payment": [
            "fee", "invoice", "payment terms", "cost", "charge", "rate", "billing",
            "hourly", "fixed fee", "retainer", "due date", "net 30"
        ],
        "Termination": [
            "cancel", "terminate", "breach", "end", "revoke", "discontinue", "cease",
            "termination for convenience", "termination for cause", "notice period"
        ],
        "Liability": [
            "indemnify", "damages", "limitation", "responsibility", "risk", "cover", "accountability",
            "cap", "liability waiver", "third-party claims", "consequential damages"
        ]
    },
    "sales": {
        "Price": [
            "price", "cost", "payment", "rate", "charge", "amount", "value",
            "pricing", "quote", "fee", "discount", "markup"
        ],
        "Delivery": [
            "shipment", "delivery", "timeline", "dispatch", "send", "transport", "arrival",
            "FOB", "shipping terms", "carrier", "lead time", "logistics"
        ],
        "Warranty": [
            "warranty", "guarantee", "defect", "assurance", "coverage", "promise", "quality",
            "merchantability", "fitness for purpose", "repair", "replace", "limited warranty"
        ],
        "Returns": [
            "refund", "return", "exchange", "credit", "replacement", "cancel", "reverse",
            "RMA", "restocking fee", "return policy", "return period"
        ]
    },
    "other": {
        "General": [
            "agreement", "party", "terms", "conditions", "contract", "deal", "understanding",
            "obligations", "rights", "governing law", "jurisdiction", "entire agreement"
        ]
    }
}


risk_terms = {
    "High": [
        "indemnify", "exclusive", "binding", "liquidated damages", "termination for cause",
        "unlimited liability", "injunction", "non-compete", "penalty", "breach of contract",
        "hold harmless", "waiver of rights", "irrevocable", "enforceable", "non-solicitation"
    ],
    "Medium": [
        "termination", "confidential", "governing law", "jurisdiction", "auto-renewal",
        "assignment", "force majeure", "compliance", "third-party", "limited liability",
        "notice period", "modification", "dispute resolution", "arbitration"
    ],
    "Low": [
        "payment", "invoice", "duration", "schedule", "definitions", "headings",
        "entire agreement", "timeline", "services", "deliverables", "fee", "scope"
    ]
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

def detect_clause_type(text, contract_type, keyword_map):
    text = text.lower()
    type_keywords = keyword_map.get(contract_type, {})
    for clause_type, keywords in type_keywords.items():
        if any(kw in text for kw in keywords):
            return clause_type
    return "Unknown"

def detect_risk_level(text, risk_terms):
    text = text.lower()
    scores = {"High": 0, "Medium": 0, "Low": 0}
    for level, keywords in risk_terms.items():
        for kw in keywords:
            if kw in text:
                scores[level] += 1
    # Return the level with the highest score
    return max(scores, key=scores.get) if any(scores.values()) else "Medium"

    if scores["High"] == scores["Medium"] == scores["Low"]:
        if "Termination" in clause_type or "Liability" in clause_type:
            return "High"
        return "Medium"

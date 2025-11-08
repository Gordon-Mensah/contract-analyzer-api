# core/config.py

CACHE_DIR = ".cache_disk"
CACHE_TTL = 60 * 60 * 24

STATE_FILE = ".session_state.json"
PERSONA_FILE = "personas.json"
FEEDBACK_FILE = "feedback_store.json"
WEIGHTS_FILE = "ranking_weights.json"
MODEL_FILE = "feedback_model.joblib"

DEFAULT_WEIGHTS = {"w_risk": 2.0, "w_sim": 1.0, "w_len": 0.5}
AUTO_TRAIN_THRESHOLD = 50
MIN_SAMPLES_FOR_MODEL = 20
MIN_POSITIVE_FOR_MODEL = 5

# Clause classification keywords
keyword_map = {
    # (Paste your full expanded keyword_map here)
}

# Risk classification keywords
risk_terms = {
    "High": [
        "penalty", "exclusive", "binding", "indemnify", "irreversible",
        "non-compete", "liquidated damages", "termination for cause", "unlimited liability", "injunction"
    ],
    "Medium": [
        "termination", "confidential", "governing law", "non-compete",
        "auto-renewal", "assignment", "jurisdiction", "force majeure", "compliance"
    ],
    "Low": [
        "notice", "duration", "payment", "invoice",
        "schedule", "timeline", "definitions", "headings", "entire agreement"
    ]
}

# Default risk level by clause type
default_risk_by_type = {
    "Liability": "High",
    "Termination": "High",
    "IP": "High",
    "Confidentiality": "Medium",
    "Restrictions": "Medium",
    "Warranty": "Medium",
    "Payment": "Low",
    "Scope": "Low",
    "Returns": "Low",
    "General": "Medium",
    "Unknown": "Medium"
}

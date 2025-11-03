# core/ranking.py

import os, json, datetime
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from core.config import FEEDBACK_FILE, MODEL_FILE, WEIGHTS_FILE, DEFAULT_WEIGHTS, AUTO_TRAIN_THRESHOLD, MIN_SAMPLES_FOR_MODEL, MIN_POSITIVE_FOR_MODEL
from core.utils import mkhash
from core.analysis import label_clause
from core.models import get_embedder
import diskcache as dc

cache_dir = os.path.join(os.getcwd(), ".cache_disk")
os.makedirs(cache_dir, exist_ok=True)
cache = dc.Cache(cache_dir)

def embed_text(text):
    key = "embed_text:" + mkhash(text)
    res = cache.get(key)
    if res is not None:
        return np.array(res, dtype=float)
    model = get_embedder()
    vec = model.encode([text])[0]
    cache.set(key, vec, expire=24 * 3600)
    return np.array(vec, dtype=float)

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def compute_risk_score(text: str) -> int:
    _, risk_level = label_clause(text)
    mapping = {"Low": 1, "Medium": 2, "High": 3}
    return mapping.get(risk_level, 1)

def score_candidate_heuristic(original, candidate):
    orig_risk = compute_risk_score(original)
    cand_risk = compute_risk_score(candidate)
    risk_delta = orig_risk - cand_risk
    sim = cosine_similarity(embed_text(original), embed_text(candidate))
    len_orig = len(original)
    len_cand = len(candidate)
    length_penalty = max(0, 1 - abs(len_cand - len_orig) / max(10, len_orig))
    w = load_weights()
    score = (w["w_risk"] * risk_delta) + (w["w_sim"] * sim) + (w["w_len"] * length_penalty)
    return float(score), {"risk_delta": risk_delta, "similarity": sim, "len_penalty": length_penalty}

def load_weights():
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_WEIGHTS.copy()

def save_weights(w):
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(w, f, ensure_ascii=False, indent=2)

def log_feedback(clause_index, original, candidate, accepted, meta, score, action):
    store = load_feedback_store()
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "clause_index": clause_index,
        "accepted": bool(accepted),
        "action": action,
        "score": float(score),
        "meta": meta,
        "original_len": len(original),
        "candidate_len": len(candidate),
        "candidate_text": candidate[:1000]
    }
    store.append(entry)
    save_feedback_store(store)
    if len(store) >= AUTO_TRAIN_THRESHOLD:
        try:
            _ = train_feedback_model(min_samples=MIN_SAMPLES_FOR_MODEL)
        except Exception:
            pass

def load_feedback_store():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_feedback_store(store):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)

def build_training_set():
    store = load_feedback_store()
    X, y = [], []
    for e in store:
        meta = e.get("meta", {})
        feat = [
            meta.get("risk_delta", 0.0),
            meta.get("similarity", 0.0),
            meta.get("len_penalty", 0.0),
            e.get("original_len", 0),
            e.get("candidate_len", 0)
        ]
        X.append(feat)
        y.append(1 if e.get("accepted") else 0)
    return np.array(X), np.array(y)

def train_feedback_model(min_samples=MIN_SAMPLES_FOR_MODEL):
    X, y = build_training_set()
    if len(X) < min_samples or sum(y) < MIN_POSITIVE_FOR_MODEL:
        return {"ok": False, "reason": "not enough data"}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_FILE)
    return {"ok": True}

def load_feedback_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

def predict_accept_prob(meta):
    clf = load_feedback_model()
    if clf is None:
        return None
    feat = np.array([
        meta.get("risk_delta", 0.0),
        meta.get("similarity", 0.0),
        meta.get("len_penalty", 0.0),
        meta.get("original_len", 0),
        meta.get("candidate_len", 0)
    ]).reshape(1, -1)
    try:
        return float(clf.predict_proba(feat)[0, 1])
    except:
        return None

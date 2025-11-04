# core/negotiation.py

import datetime, time
import streamlit as st
from core.models import get_summarizer
from core.utils import mkhash
import diskcache as dc

cache = dc.Cache(".cache_disk")

def add_turn(author, text, persona, role):
    turn_id = len(st.session_state.neg_turns) + 1
    emb = embed_text(text)
    st.session_state.neg_turns.append({
        "turn": turn_id,
        "author": author,
        "text": text,
        "persona": persona,
        "role": role,
        "embedding": emb.tolist(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    })

def summarize_clause(text):
    key = "summarize:" + mkhash(text)
    res = cache.get(key)
    if res is not None:
        return res
    summarizer = get_summarizer()
    try:
        out = summarizer(text, max_length=60, min_length=20, do_sample=False)
        s = out[0]["summary_text"] if isinstance(out, list) else out.get("summary_text", "")
    except Exception:
        s = text[:200] + "..." if len(text) > 200 else text
    cache.set(key, s, expire=24 * 3600)
    return s

def auto_negotiate_simulation(clause_text, persona, turns=3, stop_threshold=0.9, pause=0.5):
    history = []
    current = clause_text
    for t in range(turns):
        prompt = f"Rewrite clause for persona {persona}:\n\n{current}"
        candidate = summarize_clause(prompt)
        history.append(("you", candidate))
        add_turn("you", candidate, persona, "offer")
        if t > 0:
            a = embed_text(history[-2][1])
            b = embed_text(history[-1][1])
            if cosine_similarity(a, b) >= stop_threshold:
                break
        time.sleep(pause)
        sim_prompt = f"Respond as counterparty to: {candidate}"
        sim = summarize_clause(sim_prompt)
        history.append(("counterparty", sim))
        add_turn("counterparty", sim, persona, "reply")
        time.sleep(pause)
    return history

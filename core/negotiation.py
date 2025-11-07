# core/negotiation.py

import datetime, time
from core.ranking import embed_text
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
    summarizer = get_summarizer()
    if summarizer is None:
        return "⚠️ Summarizer not available."

    try:
        input_text = text[:800]
        prompt = f"Summarize this clause in plain English:\n\n{input_text}"
        out = summarizer(prompt, max_length=120, min_length=30, do_sample=False)
        return out[0]["summary_text"].strip() if isinstance(out, list) else out.get("summary_text", "").strip()
    except Exception as e:
        print(f"❌ Clause summarization failed: {e}")
        return "⚠️ Unable to summarize this clause."


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

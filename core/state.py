# core/state.py

import os, json
from core.config import STATE_FILE, PERSONA_FILE
import streamlit as st

def load_personas():
    if os.path.exists(PERSONA_FILE):
        with open(PERSONA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_session_state():
    data = {
        "contract_text": st.session_state.get("negotiation_text", ""),
        "labeled_chunks": st.session_state.get("labeled_chunks", []),
        "neg_turns": st.session_state.get("neg_turns", []),
        "neg_personas": st.session_state.get("neg_personas", {}),
        "neg_counters": st.session_state.get("neg_counters", {})
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_session_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.negotiation_text = data.get("contract_text", "")
        st.session_state.labeled_chunks = data.get("labeled_chunks", [])

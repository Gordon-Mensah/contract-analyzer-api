import streamlit as st
import requests
import tempfile
import io
import sys, os
sys.path.append(os.path.dirname(__file__))
import time
import matplotlib.pyplot as plt
import warnings


from core.config import CACHE_DIR
from core.state import load_personas, save_session_state, load_session_state
from core.analysis import (
    load_contract,
    chunk_contract,
    label_clause,
    explain_clause_risk,
    get_clause_explanation,
    explain_clause_text,
    summarize_contract
)
from core.models import get_summarizer
from core.utils import highlight_risks, format_badges
from core.negotiation import summarize_clause, add_turn
from core.export import inline_word_diff_html
from core.samples import get_sample_contract

st.set_page_config(page_title="Contract Intelligence", page_icon="📄", layout="wide")

# ---------- Candidate presentation helper ----------
def present_top_candidates_ui(original_text, clause_index, persona, style):
    st.markdown("### ✨ Suggested Counter-Proposal")
    summarizer = get_summarizer()
    base_prompt = f"Rewrite the following {st.session_state.contract_type} clause for negotiation. Persona: {persona}. Style: {style}.\n\nClause:\n{original_text}"
    try:
        out = summarizer(base_prompt, max_length=120, min_length=30, do_sample=True, top_k=50, top_p=0.95)
        text = out[0]["summary_text"].strip() if isinstance(out, list) else out.get("summary_text", "").strip()
    except Exception:
        text = original_text[:200] + "..."

    st.markdown("#### #1")
    diff_display = inline_word_diff_html(original_text, text)
    st.markdown(diff_display, unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        new_text = st.text_area(f"Edit Candidate {clause_index}_1", value=text, key=f"candidate_edit_{clause_index}_1", height=120)
    with col_b:
        if st.button(f"✅ Accept #1", key=f"accept_{clause_index}_1"):
            if "labeled_chunks" in st.session_state and 0 <= clause_index < len(st.session_state.labeled_chunks):
                st.session_state.labeled_chunks[clause_index]["text"] = new_text
                st.session_state.neg_counters[f"counter_{clause_index}"] = new_text
                add_turn("you", new_text, persona, "accepted_offer")
                st.success(f"Accepted candidate #1 for Clause {clause_index + 1}")

# ---------- Contract Type Selection ----------
st.title("📄 Smart Contract Assistant")
st.markdown("Choose the type of contract you're working with:")

contract_types = {
    "employment": "💼 Employment",
    "rental": "🏠 Rental/Lease",
    "nda": "🔒 Non-Disclosure (NDA)",
    "service": "🧰 Service Agreement",
    "sales": "🛒 Sales/Purchase",
    "other": "📁 Other"
}

selected_type = st.selectbox("Contract Type", list(contract_types.values()))
st.session_state.contract_type = [k for k, v in contract_types.items() if v == selected_type][0]

if st.button("📄 Load Sample Contract"):
    sample_text = get_sample_contract(st.session_state.contract_type)
    st.session_state.negotiation_text = sample_text
    st.session_state.contract_loaded = True
    st.success(f"Sample {contract_types[st.session_state.contract_type]} contract loaded.")

# ---------- Session defaults ----------
_defaults = {
    "neg_personas": {},
    "neg_counters": {},
    "neg_simulated": {},
    "neg_turns": [],
    "labeled_chunks": [],
    "negotiation_text": "",
    "saved_personas": load_personas(),
    "contract_loaded": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
load_session_state()

# ---------- Sidebar ----------
st.sidebar.title("📄 Contract Intelligence")
st.sidebar.markdown("Build smarter contracts with zero budget")

with st.sidebar.expander("📁 Upload Contract"):
    uploaded_file = st.file_uploader("Choose a contract file", type=["pdf", "docx", "txt"])
    if uploaded_file:
        text = load_contract(uploaded_file)
        st.session_state.negotiation_text = text
        st.session_state.contract_loaded = True

with st.sidebar.expander("🌐 Import from Link"):
    contract_url = st.text_input("Paste contract URL (PDF, DOCX, or TXT)")
    if contract_url:
        try:
            response = requests.get(contract_url)
            if response.status_code == 200:
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(response.content)
                temp_file.flush()
                temp_file.seek(0)
                text = load_contract(temp_file)
                st.session_state.negotiation_text = text
                st.success("Contract imported from link.")
            else:
                st.warning(f"Failed to fetch document. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error fetching document: {e}")

learn_mode = st.sidebar.checkbox("🧠 Learn as You Go", value=True)
summarize_enabled = st.sidebar.checkbox("Summarize clauses", value=True)

with st.sidebar.expander("🧠 Persona Settings"):
    persona = st.text_input("Persona name", value=st.session_state.neg_personas.get("default", {}).get("persona", "Startup Founder"))
    style = st.selectbox("Rewrite style", ["Plain English", "Legalese", "Assertive", "Concise", "Friendly"])
    st.session_state.neg_personas["default"] = {"persona": persona, "style": style}

if st.sidebar.button("💾 Save Session"):
    save_session_state()
if st.sidebar.button("📂 Load Session"):
    load_session_state()
if st.sidebar.button("🔁 Clear Counters"):
    st.session_state.neg_counters = {}
    st.success("Cleared accepted counters.")

# ---------- Main UI ----------
if st.session_state.contract_loaded:
    st.subheader("📜 Original Contract Text")
    st.text_area("Contract", value=st.session_state.negotiation_text, height=200)

if st.session_state.contract_loaded:
    st.subheader("🧾 Contract Summary")
    with st.spinner("Generating contract summary..."):
        summary = summarize_contract(st.session_state.negotiation_text)
    st.markdown(summary)



if st.session_state.negotiation_text and st.button("🔍 Analyze Clauses"):
    start = time.time()
    chunks = chunk_contract(st.session_state.negotiation_text)
    chunks = chunks[:10000]
    labeled = []
    for i, chunk in enumerate(chunks):
        clause_type, risk_level = label_clause(chunk, st.session_state.contract_type)
        summary = summarize_clause(chunk) if summarize_enabled else ""
        labeled.append({
            "id": i,
            "text": chunk,
            "type": clause_type,
            "risk": risk_level,
            "summary": summary,
            "translated": ""
        })
    st.session_state.labeled_chunks = labeled
    st.success(f"{len(labeled)} clauses analyzed.")
    st.write(f"⏱️ Clause analysis took {time.time() - start:.2f} seconds")

    risk_counts = {"High": 0, "Medium": 0, "Low": 0}
    for c in labeled:
        if c["risk"] in risk_counts:
            risk_counts[c["risk"]] += 1

    fig, ax = plt.subplots()
    ax.bar(risk_counts.keys(), risk_counts.values(), color=["red", "orange", "green"])
    ax.set_title("Clause Risk Summary")
    ax.set_ylabel("Number of Clauses")
    st.pyplot(fig)

# ---------- Clause Review ----------
if st.session_state.labeled_chunks:
    st.subheader("🧩 Clause Review")

    risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])
    type_filter = st.selectbox("Filter by Clause Type", ["All"] + sorted(set(c["type"] for c in st.session_state.labeled_chunks)))

    filtered_clauses = [
        c for c in st.session_state.labeled_chunks
        if (risk_filter == "All" or c["risk"] == risk_filter)
        and (type_filter == "All" or c["type"] == type_filter)
    ]

    for key in list(st.session_state.keys()):
        if key.startswith("candidate_edit_") or key.startswith("learn_check_"):
            del st.session_state[key]

        for i, clause in enumerate(filtered_clauses):
            with st.expander(f"Clause {i+1}: {format_badges(clause['type'], clause['risk'])}"):
                st.markdown(highlight_risks(clause["text"]), unsafe_allow_html=True)

        if clause["summary"]:
            st.markdown(f"**Summary:** {clause['summary']}")

        risk_note = explain_clause_risk(clause["text"], clause["type"], clause["risk"])
        if risk_note:
            st.markdown(f"**Why this clause matters:** {risk_note}")

        # Civilian-friendly suggestions
        if clause["risk"] in ["High", "Medium"]:
            st.markdown("💬 **What you could ask:**")
            if clause["type"] == "Termination":
                st.markdown("- Can we extend the notice period?")
                st.markdown("- Can termination only happen with cause?")
            elif clause["type"] == "Payment":
                st.markdown("- Can we clarify late fees or payment schedule?")
            elif clause["type"] == "Confidentiality":
                st.markdown("- Can we limit how long confidentiality lasts?")
            elif clause["type"] == "Liability":
                st.markdown("- Can we cap the damages or limit responsibility?")
            elif clause["type"] == "IP":
                st.markdown("- Can we clarify who owns the work created?")
            elif clause["type"] == "Scope":
                st.markdown("- Can we define exactly what services are included?")

        # Learn mode explanation
        if learn_mode:
            st.markdown("**🧠 Do you understand this clause?**")
            understanding = st.radio(
                f"Understanding Check {clause['id']}",
                ["Yes", "Not sure", "No"],
                key=f"learn_check_{clause['id']}"
            )
            if understanding == "No":
                explanation = explain_clause_text(clause["text"])
                if explanation:
                    st.info(f"🧾 **What this clause means in simple terms:**\n\n{explanation}")
                else:
                    st.warning("⚠️ Sorry, I couldn't simplify this clause at the moment.")

        present_top_candidates_ui(clause["text"], clause["id"], persona, style)


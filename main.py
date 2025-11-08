import streamlit as st
import requests
import tempfile
import io
import sys, os
import traceback
import time
import matplotlib.pyplot as plt
sys.path.append("core")


from core.config import CACHE_DIR
from core.state import load_personas, save_session_state, load_session_state
from core.analysis import (
    load_contract,
    chunk_contract,
    label_clause,
    explain_clause_risk,
    get_clause_explanation,
    explain_clause_text
      )

from core.utils import highlight_risks, format_badges
from core.samples import get_sample_contract

st.set_page_config(page_title="Contract Intelligence", page_icon="📄", layout="wide")

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
            if not text or not isinstance(text, str):
                st.error("🚨 Loaded contract is empty or invalid.")
                st.stop()
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

explain_simple = st.sidebar.checkbox("🧒 Simplify explanations", value=False)
 
with st.sidebar.expander("🧠 Persona Settings"):
        persona = st.text_input("Persona name", value=st.session_state.neg_personas.get("default", {}).get("persona", "Startup Founder"))
        style = st.selectbox("Rewrite style", ["Plain English", "Legalese", "Assertive", "Concise", "Friendly"])
        st.session_state.neg_personas["default"] = {"persona": persona, "style": style}

    # ---------- Main UI ----------
if st.session_state.contract_loaded:
        st.subheader("📜 Original Contract Text")
        st.text_area("Contract", value=st.session_state.negotiation_text, height=200)

if st.session_state.negotiation_text and st.button("🔍 Analyze Clauses"):
        start = time.time()
        try:
            chunks = chunk_contract(st.session_state.negotiation_text)
        except Exception as e:
            st.error(f"🚨 Failed to chunk contract: {e}")
            chunks = []

        chunks = chunks[:300]
        labeled = []

        st.write(f"🔍 Analyzing {len(chunks)} clauses...")

        for i, chunk in enumerate(chunks[:1000]):
            clause_type, risk_level, summary = "Unknown", "Medium", ""
            try:
                if not chunk or not isinstance(chunk, str):
                    raise ValueError("Empty or invalid clause text")

                summary = ""  # Skipped for lightweight mode


            except Exception as e:
                st.warning(f"⚠️ Error analyzing clause {i+1}: {e}")
                st.text(traceback.format_exc())

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

        total = sum(risk_counts.values())
        score = (
            risk_counts["Low"] * 1 +
            risk_counts["Medium"] * 0.5 +
            risk_counts["High"] * 0
        ) / max(total, 1)
        confidence = "🔴 Low" if score < 0.4 else "🟠 Medium" if score < 0.7 else "🟢 High"
        st.metric("📊 Contract Confidence Score", confidence)

        # ---------- Risk Report ----------
        report_lines = []
        report_lines.append(f"📄 Contract Risk Report")
        report_lines.append(f"Contract Type: {contract_types[st.session_state.contract_type]}")
        report_lines.append(f"Total Clauses Analyzed: {len(labeled)}")
        report_lines.append(f"Risk Breakdown:")
        report_lines.append(f"- High Risk: {risk_counts['High']}")
        report_lines.append(f"- Medium Risk: {risk_counts['Medium']}")
        report_lines.append(f"- Low Risk: {risk_counts['Low']}")

        common_types = {}
        for c in labeled:
            t = c["type"]
            if t not in common_types:
                common_types[t] = 0
            common_types[t] += 1

        top_types = sorted(common_types.items(), key=lambda x: x[1], reverse=True)[:3]
        report_lines.append("Most Common Clause Types: " + ", ".join(t[0] for t in top_types))
        report_lines.append(f"Overall Confidence Score: {confidence}")

        top_clauses = [c for c in labeled if c["risk"] in ["High", "Medium"]][:5]
        report_lines.append("Top Clauses to Review:")
        for c in top_clauses:
            report_lines.append(f"- Clause {c['id']+1} — {c['type']} — {c['risk']} Risk")

        report_text = "\n".join(report_lines)

        report_lines.append("Negotiation Prompts:")
        for c in top_clauses:
            prompts = []
            if c["type"] == "Termination":
                prompts = [
                    "Can we extend the notice period?",
                    "Can termination only happen with cause?"
                ]
            elif c["type"] == "Payment":
                prompts = [
                    "Can we clarify late fees or payment schedule?"
                ]
            elif c["type"] == "Confidentiality":
                prompts = [
                    "Can we limit how long confidentiality lasts?"
                ]
            elif c["type"] == "Liability":
                prompts = [
                    "Can we cap the damages or limit responsibility?"
                ]
            elif c["type"] == "IP":
                prompts = [
                    "Can we clarify who owns the work created?"
                ]
            elif c["type"] == "Scope":
                prompts = [
                    "Can we define exactly what services are included?"
                ]
            if prompts:
                report_lines.append(f"- Clause {c['id']+1} ({c['type']}):")
                for p in prompts:
                    report_lines.append(f"  • {p}")


        st.download_button(
            label="📥 Download Risk Report",
            data=report_text,
            file_name="contract_risk_report.txt",
            mime="text/plain"
        )


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

        for i, clause in enumerate(filtered_clauses):
            with st.expander(f"Clause {i+1}: {format_badges(clause['type'], clause['risk'])}"):
                st.markdown(highlight_risks(clause["text"]), unsafe_allow_html=True)

                if clause["summary"]:
                    st.markdown(f"**Summary:** {clause['summary']}")

                risk_note = explain_clause_risk(clause["text"], clause["type"], clause["risk"])
                if risk_note:
                    st.markdown(f"**Why this clause matters:** {risk_note}")

                # 🧒 Simplified explanation
                if explain_simple:
                    explanation = explain_clause_text(clause["text"])
                    if explanation:
                        st.info(f"🧾 **Simple Explanation:**\n\n{explanation}")

                # Civilian-friendly suggestions
            if clause["risk"] in ["High", "Medium"]:
                with st.expander("💬 What you could ask"):
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

                    

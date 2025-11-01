from fastapi import FastAPI, UploadFile
from core.analysis import label_clause, explain_clause_risk, get_clause_explanation

app = FastAPI()

@app.post("/analyze")
async def analyze_clause(file: UploadFile, contract_type: str = "nda"):
    text = (await file.read()).decode()
    clause_type, risk = label_clause(text, contract_type)
    explanation = get_clause_explanation(clause_type)
    risk_note = explain_clause_risk(text, clause_type, risk)
    return {
        "clause_type": clause_type,
        "risk_level": risk,
        "explanation": explanation,
        "risk_note": risk_note
    }

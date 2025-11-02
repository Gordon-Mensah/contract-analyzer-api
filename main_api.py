from fastapi import FastAPI, UploadFile
from core.analysis import label_clause, explain_clause_risk, get_clause_explanation
from fastapi.middleware.cors import CORSMiddleware
from pdfminer.high_level import extract_text

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000"],  # or your deployed frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_clause(file: UploadFile, contract_type: str = "nda"):
    contents = await file.read()
    text = extract_text(BytesIO(contents))  # Extract text from PDF

    clause_type, risk = label_clause(text, contract_type)
    explanation = get_clause_explanation(clause_type)
    risk_note = explain_clause_risk(text, clause_type, risk)

    return {
        "clause_type": clause_type,
        "risk_level": risk,
        "explanation": explanation,
        "risk_note": risk_note
    }

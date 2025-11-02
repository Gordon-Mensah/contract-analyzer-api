<<<<<<< HEAD
from fastapi import FastAPI, UploadFile
from core.analysis import label_clause, explain_clause_risk, get_clause_explanation
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
=======
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

>>>>>>> 4aaee57cc5d2991792529ccec2efc567e48ceed0

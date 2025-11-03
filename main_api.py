from fastapi import FastAPI, UploadFile
from core.analysis import label_clause, explain_clause_risk, get_clause_explanation
from fastapi.middleware.cors import CORSMiddleware
from pdfminer.high_level import extract_text
import pdfplumber
from io import BytesIO


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
    try:
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")

        if file.content_type != "application/pdf":
            return {"detail": "Uploaded file is not a PDF"}

        with pdfplumber.open(BytesIO(contents)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        print(f"Extracted text: {text[:500]}")

        clause_type, risk = label_clause(text, contract_type)
        explanation = get_clause_explanation(clause_type)
        risk_note = explain_clause_risk(text, clause_type, risk)

        return {
            "clause_type": clause_type,
            "risk_level": risk,
            "explanation": explanation,
            "risk_note": risk_note
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"detail": f"Error processing file: {str(e)}"}

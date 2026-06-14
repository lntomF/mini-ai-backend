from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "For personal AI Backend is running"}

@app.post("/analyze")
def analyze_text(request: TextRequest):
    words = request.text.split()
    return {
        "original_text": request.text,
        "word_count": len(words),
        "char_count": len(request.text),
        "summary": f"This text has {len(words)} words."
    }
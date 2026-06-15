import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(title="Mini AI Backend API with DeepSeek")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


class TextRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "Mini AI Backend with DeepSeek is running"}


@app.post("/analyze")
def analyze_text(request: TextRequest):
    words = request.text.split()

    return {
        "original_text": request.text,
        "word_count": len(words),
        "char_count": len(request.text),
        "summary": f"This text has {len(words)} words."
    }


@app.post("/chat")
def chat(request: ChatRequest):
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY is missing. Please set it in your .env file."
        )

    try:
        response = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Explain things clearly and simply."
                },
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            stream=False
        )

        return {
            "reply": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DeepSeek request failed: {str(e)}"
        )
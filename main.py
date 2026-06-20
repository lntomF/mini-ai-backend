import os
import uuid
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

load_dotenv()

app = FastAPI(title="Mini RAG Assistant with DeepSeek")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

CHROMA_DIR = Path("chroma_db")
CHROMA_COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
VECTOR_DISTANCE_THRESHOLD = 1.5
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DOCUMENT_STORE: dict[str, dict] = {}

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
vector_collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class AskDocumentRequest(BaseModel):
    question: str = Field(..., min_length=1)
    doc_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)


class VectorSearchDocumentRequest(BaseModel):
    query: str = Field(..., min_length=1)
    doc_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Mini RAG Assistant is running",
        "endpoints": [
            "GET /",
            "GET /health",
            "POST /chat",
            "POST /upload-document",
            "POST /vector-search-document",
            "POST /ask-document-vector",
        ],
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "mini-rag-assistant",
        "document_count": len(DOCUMENT_STORE),
        "vector_count": vector_collection.count(),
    }


@app.post("/chat")
def chat(request: ChatRequest):
    check_api_key()

    try:
        completion = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Explain things clearly and simply.",
                },
                {"role": "user", "content": request.message},
            ],
            stream=False,
        )
        return {"reply": completion.choices[0].message.content}

    except Exception:
        raise HTTPException(status_code=500, detail="Chat service error.")


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    allowed_extensions = {".txt", ".md"}
    original_filename = Path(file.filename or "document.txt").name
    file_suffix = Path(original_filename).suffix.lower()

    if file_suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .md files are supported for now.",
        )

    doc_id = str(uuid.uuid4())
    saved_filename = f"{doc_id}{file_suffix}"
    file_path = UPLOAD_DIR / saved_filename

    try:
        content_bytes = await file.read()

        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be encoded in UTF-8.")

        chunks = chunk_text(text)

        if not chunks:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        file_path.write_text(text, encoding="utf-8")

        DOCUMENT_STORE[doc_id] = {
            "filename": original_filename,
            "saved_path": str(file_path),
            "chunk_count": len(chunks),
        }

        chunk_ids = [f"{doc_id}_{index}" for index in range(len(chunks))]
        chunk_metadatas = [
            {
                "doc_id": doc_id,
                "filename": original_filename,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]
        chunk_embeddings = embedding_model.encode(chunks).tolist()

        vector_collection.add(
            ids=chunk_ids,
            documents=chunks,
            metadatas=chunk_metadatas,
            embeddings=chunk_embeddings,
        )

        return {
            "doc_id": doc_id,
            "filename": original_filename,
            "char_count": len(text),
            "chunk_count": len(chunks),
            "chunks_preview": chunks[:3],
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(status_code=500, detail="Document upload failed.")


@app.post("/vector-search-document")
def vector_search_document(request: VectorSearchDocumentRequest):
    if vector_collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No document embeddings found. Please upload a document first.",
        )

    results = vector_search(
        query=request.query,
        top_k=request.top_k,
        doc_id=request.doc_id,
    )

    return {
        "query": request.query,
        "top_k": request.top_k,
        "result_count": len(results),
        "results": results,
    }


@app.post("/ask-document-vector")
def ask_document_vector(request: AskDocumentRequest):
    check_api_key()

    if vector_collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No document embeddings found. Please upload a document first.",
        )

    top_results = vector_search(
        query=request.question,
        top_k=request.top_k,
        doc_id=request.doc_id,
    )

    if not top_results:
        return {
            "question": request.question,
            "answer": "I don't know based on the uploaded documents.",
            "citations": [],
            "retrieved_chunks": [],
        }

    context = "\n\n".join(
        f"[Source {index}] filename: {result['filename']}, "
        f"chunk_index: {result['chunk_index']}\n{result['text']}"
        for index, result in enumerate(top_results, start=1)
    )

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question using only the provided context.

Rules:
- If the answer is not in the context, say: "I don't know based on the uploaded documents."
- Do not use outside knowledge.
- Keep the answer clear and concise.
- Include citation markers like [Source 1] when using information from the context.
"""

    user_prompt = f"""
Question:
{request.question}

Context:
{context}
"""

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )

        return {
            "question": request.question,
            "answer": response.choices[0].message.content,
            "retrieval_method": "vector_search",
            "citations": [
                {
                    "source": f"Source {index}",
                    "doc_id": result["doc_id"],
                    "filename": result["filename"],
                    "chunk_index": result["chunk_index"],
                    "distance": result["distance"],
                }
                for index, result in enumerate(top_results, start=1)
            ],
            "retrieved_chunks": top_results,
        }

    except Exception:
        raise HTTPException(status_code=500, detail="Vector document QA failed.")


def check_api_key():
    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY is missing. Please set it in your .env file.",
        )


def vector_search(
    query: str,
    top_k: int = 3,
    doc_id: str | None = None,
    distance_threshold: float = VECTOR_DISTANCE_THRESHOLD,
):
    query_embedding = embedding_model.encode([query]).tolist()
    where_filter = {"doc_id": doc_id} if doc_id else None

    results = vector_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    search_results = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        if distance > distance_threshold:
            continue

        search_results.append(
            {
                "doc_id": metadata.get("doc_id"),
                "filename": metadata.get("filename"),
                "chunk_index": metadata.get("chunk_index"),
                "distance": distance,
                "text": document,
            }
        )

    return search_results


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks

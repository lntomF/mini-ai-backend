import ast
import json
import operator
import os
import re
from typing import Any
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from openai import OpenAI
from pydantic import BaseModel, Field
import uuid
from pathlib import Path

load_dotenv()

app = FastAPI(title="Mini AI Backend with DeepSeek")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

VECTOR_DISTANCE_THRESHOLD = 1.5

RAG_EVAL_CASES = [
    {
        "question": "How does RAG reduce hallucination?",
        "should_answer": True,
        "expected_keywords": ["hallucination", "context"],
    },
    {
        "question": "How can RAG avoid making things up?",
        "should_answer": True,
        "expected_keywords": ["context"],
    },
    {
        "question": "Who is the CEO of NVIDIA?",
        "should_answer": False,
        "expected_keywords": [],
    },
]

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
DOCUMENT_STORE = {}
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "who", "what", "when", "where", "why", "how",
    "of", "to", "in", "on", "for", "with", "and", "or",
    "does", "do", "did", "be", "by", "from",
    "can", "could", "would", "should", "may", "might"
}
CHROMA_DIR = Path("chroma_db")
CHROMA_COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

vector_collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME
)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)

class ClassifyRequest(BaseModel):
    message: str = Field(..., min_length=1)

class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1)

class StudyPlanRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    days: int = Field(..., ge=1, le=30)

class SearchDocumentRequest(BaseModel):
    query: str = Field(..., min_length=1)
    doc_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)

class AskDocumentRequest(BaseModel):
    question: str = Field(..., min_length=1)
    doc_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)

class VectorSearchDocumentRequest(BaseModel):
    query: str = Field(..., min_length=1)
    doc_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)

@app.get("/")
def home():
    return {
        "message": "Mini AI Backend with DeepSeek is running",
        "endpoints": [
            "GET /",
            "POST /analyze",
            "POST /chat",
            "POST /classify",
            "POST /agent",
            "POST /upload-document"
        ]
    }

@app.post("/analyze")
def analyze_text(request: TextRequest):
    words = request.text.split()

    return {
        "original_text": request.text,
        "word_count": len(words),
        "char_count": len(request.text),
        "summary": f"This text has {len(words)} words and {len(request.text)} characters."
    }

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    allowed_extensions = [".txt", ".md"]

    file_suffix = Path(file.filename).suffix.lower()

    if file_suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .md files are supported for now."
        )

    doc_id = str(uuid.uuid4())
    saved_filename = f"{doc_id}_{file.filename}"
    file_path = UPLOAD_DIR / saved_filename

    try:
        content_bytes = await file.read()

        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File must be encoded in UTF-8."
            )

        file_path.write_text(text, encoding="utf-8")

        chunks = chunk_text(text)

        DOCUMENT_STORE[doc_id] = {
        "filename": file.filename,
        "saved_path": str(file_path),
        "text": text,
        "chunks": chunks
        }
        
        chunk_ids = [
            f"{doc_id}_{index}"
            for index in range(len(chunks))
        ]

        chunk_metadatas = [
            {
                "doc_id": doc_id,
                "filename": file.filename,
                "chunk_index": index
            }
            for index in range(len(chunks))
        ]

        chunk_embeddings = embedding_model.encode(chunks).tolist()

        vector_collection.add(
            ids=chunk_ids,
            documents=chunks,
            metadatas=chunk_metadatas,
            embeddings=chunk_embeddings
        )


        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "saved_path": str(file_path),
            "char_count": len(text),
            "chunk_count": len(chunks),
            "chunks_preview": chunks[:3]
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {str(e)}"
        )

@app.post("/ask-document")
def ask_document(request: AskDocumentRequest):
    check_api_key()

    if not DOCUMENT_STORE:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet."
        )

    search_targets = {}

    if request.doc_id:
        if request.doc_id not in DOCUMENT_STORE:
            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )
        search_targets[request.doc_id] = DOCUMENT_STORE[request.doc_id]
    else:
        search_targets = DOCUMENT_STORE

    all_results = []

    for doc_id, doc in search_targets.items():
        results = keyword_search(
            query=request.question,
            chunks=doc["chunks"],
            top_k=request.top_k
        )

        for result in results:
            all_results.append({
            "doc_id": doc_id,
            "filename": doc["filename"],
            "chunk_index": result["chunk_index"],
            "score": result["score"],
            "matched_words": result["matched_words"],
            "text": result["text"]
            })

    all_results.sort(key=lambda item: item["score"], reverse=True)
    top_results = all_results[:request.top_k]

    if not top_results:
        return {
            "question": request.question,
            "answer": "I don't know based on the uploaded documents.",
            "citations": [],
            "retrieved_chunks": []
        }

    context_blocks = []

    for index, result in enumerate(top_results, start=1):
        context_blocks.append(
            f"[Source {index}] "
            f"filename: {result['filename']}, "
            f"chunk_index: {result['chunk_index']}\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_blocks)

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
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            stream=False,
        )

        return {
            "question": request.question,
            "answer": response.choices[0].message.content,
            "citations": [
                {
                    "source": f"Source {index}",
                    "doc_id": result["doc_id"],
                    "filename": result["filename"],
                    "chunk_index": result["chunk_index"],
                    "score": result["score"]
                }
                for index, result in enumerate(top_results, start=1)
            ],
            "retrieved_chunks": top_results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document QA failed: {str(e)}"
        )

@app.post("/vector-search-document")
def vector_search_document(request: VectorSearchDocumentRequest):
    if vector_collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No document embeddings found. Please upload a document first."
        )

    if request.doc_id and request.doc_id not in DOCUMENT_STORE:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    results = vector_search(
        query=request.query,
        top_k=request.top_k,
        doc_id=request.doc_id
    )

    return {
        "query": request.query,
        "top_k": request.top_k,
        "result_count": len(results),
        "results": results
    }

@app.post("/chat")
def chat(request: ChatRequest):
    check_api_key()

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
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
            stream=False,
        )

        return {
            "reply": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DeepSeek chat request failed: {str(e)}"
        )


@app.post("/classify")
def classify_message(request: ClassifyRequest):
    check_api_key()

    system_prompt = """
You are an intent classification engine.

Classify the user's message into exactly one intent.

Available intents:
- qa: user asks a question
- summarization: user wants to summarize content
- translation: user wants translation
- coding: user asks for code or debugging
- writing: user wants writing or rewriting
- calculation: user asks for math calculation
- text_analysis: user wants text statistics or analysis
- unknown: unclear intent

You must output valid json only.

Example JSON output:
{
  "intent": "summarization",
  "confidence": 0.92,
  "reason": "The user asks to summarize a document."
}
"""

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            stream=False,
        )

        raw_content = response.choices[0].message.content

        if not raw_content:
            raise HTTPException(
                status_code=500,
                detail="Model returned empty content."
            )

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Model did not return valid JSON.",
                    "raw_content": raw_content
                }
            )

        return parsed

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DeepSeek structured output failed: {str(e)}"
        )

@app.post("/ask-document-vector")
def ask_document_vector(request: AskDocumentRequest):
    check_api_key()

    if vector_collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No document embeddings found. Please upload a document first."
        )

    if request.doc_id and request.doc_id not in DOCUMENT_STORE:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    top_results = vector_search(
        query=request.question,
        top_k=request.top_k,
        doc_id=request.doc_id
    )

    if not top_results:
        return {
            "question": request.question,
            "answer": "I don't know based on the uploaded documents.",
            "citations": [],
            "retrieved_chunks": []
        }

    context_blocks = []

    for index, result in enumerate(top_results, start=1):
        context_blocks.append(
            f"[Source {index}] "
            f"filename: {result['filename']}, "
            f"chunk_index: {result['chunk_index']}\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_blocks)

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
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
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
                    "distance": result["distance"]
                }
                for index, result in enumerate(top_results, start=1)
            ],
            "retrieved_chunks": top_results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vector document QA failed: {str(e)}"
        )

@app.post("/study-plan")
def create_study_plan(request: StudyPlanRequest):
    check_api_key()

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a practical AI internship coach. Create clear and actionable study plans."
                },
                {
                    "role": "user",
                    "content": f"请为目标：{request.goal}，制定一个 {request.days} 天的学习计划。要求每天都有具体任务和产出物。"
                }
            ],
            stream=False,
        )

        return {
            "goal": request.goal,
            "days": request.days,
            "plan": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Study plan generation failed: {str(e)}"
        )

@app.post("/search-document")
def search_document(request: SearchDocumentRequest):
    if not DOCUMENT_STORE:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet."
        )

    search_targets = {}

    if request.doc_id:
        if request.doc_id not in DOCUMENT_STORE:
            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )
        search_targets[request.doc_id] = DOCUMENT_STORE[request.doc_id]
    else:
        search_targets = DOCUMENT_STORE

    all_results = []

    for doc_id, doc in search_targets.items():
        results = keyword_search(
            query=request.query,
            chunks=doc["chunks"],
            top_k=request.top_k
        )

        for result in results:
            all_results.append({
            "doc_id": doc_id,
            "filename": doc["filename"],
            "chunk_index": result["chunk_index"],
            "score": result["score"],
            "matched_words": result["matched_words"],
            "text": result["text"]
            })
    all_results.sort(key=lambda item: item["score"], reverse=True)

    return {
        "query": request.query,
        "top_k": request.top_k,
        "result_count": len(all_results[:request.top_k]),
        "results": all_results[:request.top_k]
    }

@app.post("/agent")
def run_agent(request: AgentRequest):
    check_api_key()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a tool-using AI agent. "
                "When the user asks for arithmetic calculation, use calculator. "
                "When the user asks for word or character counting, use word_counter. "
                "If no tool is needed, answer directly. "
                "After receiving tool results, answer clearly and briefly."
            )
        },
        {
            "role": "user",
            "content": request.message
        }
    ]

    try:
        first_response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=AGENT_TOOLS,
            tool_choice="auto",
            stream=False,
        )

        assistant_message = first_response.choices[0].message

        if not assistant_message.tool_calls:
            return {
                "mode": "direct_answer",
                "reply": assistant_message.content,
                "tool_results": []
            }

        assistant_tool_calls = []

        for tool_call in assistant_message.tool_calls:
            assistant_tool_calls.append({
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })

        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": assistant_tool_calls
        })

        tool_results = []

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name

            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}

            result = execute_tool(tool_name, tool_args)
            tool_results.append(result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

        final_response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            stream=False,
        )

        return {
            "mode": "tool_used",
            "reply": final_response.choices[0].message.content,
            "tool_results": tool_results
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed: {str(e)}"
        )
    
@app.post("/evaluate-rag")
def evaluate_rag():
    check_api_key()

    if vector_collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No document embeddings found. Please upload a document first."
        )

    eval_results = []

    for case in RAG_EVAL_CASES:
        question = case["question"]

        top_results = vector_search(
            query=question,
            top_k=3
        )

        if not top_results:
            answer = "I don't know based on the uploaded documents."
            citations = []
        else:
            context_blocks = []

            for index, result in enumerate(top_results, start=1):
                context_blocks.append(
                    f"[Source {index}] "
                    f"filename: {result['filename']}, "
                    f"chunk_index: {result['chunk_index']}\n"
                    f"{result['text']}"
                )

            context = "\n\n".join(context_blocks)

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
{question}

Context:
{context}
"""

            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                stream=False,
            )

            answer = response.choices[0].message.content

            citations = [
                {
                    "source": f"Source {index}",
                    "doc_id": result["doc_id"],
                    "filename": result["filename"],
                    "chunk_index": result["chunk_index"],
                    "distance": result["distance"]
                }
                for index, result in enumerate(top_results, start=1)
            ]

        evaluation = evaluate_answer(
            answer=answer,
            should_answer=case["should_answer"],
            expected_keywords=case["expected_keywords"]
        )

        eval_results.append({
            "question": question,
            "should_answer": case["should_answer"],
            "answer": answer,
            "citations": citations,
            "passed": evaluation["passed"],
            "reason": evaluation["reason"]
        })

    total = len(eval_results)
    passed = sum(1 for item in eval_results if item["passed"])

    return {
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": round(passed / total, 2),
        "results": eval_results
    }

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a math expression. Use this when the user asks for arithmetic calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, for example: 23 * 19 + 7"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "word_counter",
            "description": "Count words and characters in a given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze."
                    }
                },
                "required": ["text"]
            }
        }
    }
]

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def check_api_key():
    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY is missing. Please set it in your .env file."
        )

def vector_search(
    query: str,
    top_k: int = 3,
    doc_id: str | None = None,
    distance_threshold: float = VECTOR_DISTANCE_THRESHOLD
):
    query_embedding = embedding_model.encode([query]).tolist()

    where_filter = None

    if doc_id:
        where_filter = {"doc_id": doc_id}

    results = vector_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    search_results = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        if distance > distance_threshold:
            continue

        search_results.append({
            "doc_id": metadata.get("doc_id"),
            "filename": metadata.get("filename"),
            "chunk_index": metadata.get("chunk_index"),
            "distance": distance,
            "text": document
        })

    return search_results

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Split long text into overlapping chunks.
    This is the first step of a RAG pipeline.
    """

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

        if start >= len(text):
            break

    return chunks

def keyword_search(query: str, chunks: list[str], top_k: int = 3):
    """
    Simple keyword-based retrieval.

    This version removes common stopwords so that words like
    'is', 'the', 'who' do not create false matches.
    """

    query_words = re.findall(r"\b\w+\b", query.lower())

    important_words = [
        word for word in query_words
        if word not in STOPWORDS and len(word) > 2
    ]

    if not important_words:
        return []

    results = []

    for index, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = 0
        matched_words = []

        for word in important_words:
            if word in chunk_lower:
                score += 1
                matched_words.append(word)

        if score > 0:
            results.append({
                "chunk_index": index,
                "score": score,
                "matched_words": matched_words,
                "text": chunk
            })

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]

def execute_tool(tool_name: str, arguments: dict[str, Any]):
    if tool_name == "calculator":
        expression = arguments.get("expression", "")

        if not expression:
            raise HTTPException(
                status_code=400,
                detail="calculator requires expression."
            )

        result = safe_calculate(expression)

        return {
            "tool": "calculator",
            "expression": expression,
            "result": result
        }

    if tool_name == "word_counter":
        text = arguments.get("text", "")
        words = text.split()

        return {
            "tool": "word_counter",
            "text": text,
            "word_count": len(words),
            "char_count": len(text)
        }

    raise HTTPException(
        status_code=400,
        detail=f"Unknown tool: {tool_name}"
    )

def safe_calculate(expression: str):
    """
    Safer calculator.
    Do not use eval().
    Only supports numbers and basic math operators.
    """

    if len(expression) > 100:
        raise HTTPException(
            status_code=400,
            detail="Expression is too long."
        )

    try:
        tree = ast.parse(expression, mode="eval")
        result = eval_ast_node(tree.body)

        if abs(result) > 1_000_000_000:
            raise HTTPException(
                status_code=400,
                detail="Calculation result is too large."
            )

        return result

    except ZeroDivisionError:
        raise HTTPException(
            status_code=400,
            detail="Division by zero is not allowed."
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid math expression: {str(e)}"
        )

def eval_ast_node(node):
    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        left = eval_ast_node(node.left)
        right = eval_ast_node(node.right)
        op_type = type(node.op)

        if op_type not in ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed.")

        return ALLOWED_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operand = eval_ast_node(node.operand)
        op_type = type(node.op)

        if op_type not in ALLOWED_OPERATORS:
            raise ValueError("Unary operator not allowed.")

        return ALLOWED_OPERATORS[op_type](operand)

    raise ValueError("Invalid expression.")

def evaluate_answer(answer: str, should_answer: bool, expected_keywords: list[str]):
    answer_lower = answer.lower()

    if not should_answer:
        passed = "i don't know" in answer_lower
        return {
            "passed": passed,
            "reason": "Expected refusal answer."
        }

    missing_keywords = []

    for keyword in expected_keywords:
        if keyword.lower() not in answer_lower:
            missing_keywords.append(keyword)

    passed = len(missing_keywords) == 0

    return {
        "passed": passed,
        "reason": (
            "Answer contains expected keywords."
            if passed
            else f"Missing keywords: {missing_keywords}"
        )
    }
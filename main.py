import os
import uuid
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from services.blog_indexer import sync_all_blog_posts


load_dotenv()


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="Mini RAG Assistant with DeepSeek",
    version="1.0.0",
)


# =========================================================
# 环境变量
# =========================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-chat",
)

BLOG_BASE_URL = os.getenv(
    "BLOG_BASE_URL",
    "http://localhost:4321",
)

BLOG_URL_PREFIX = os.getenv(
    "BLOG_URL_PREFIX",
    "/blog",
)


# =========================================================
# DeepSeek 客户端
# =========================================================

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


# =========================================================
# 本地目录
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "rag_documents"


# =========================================================
# RAG 配置
# =========================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Chroma 默认通常使用距离，越小越相似
VECTOR_DISTANCE_THRESHOLD = 1.5

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# 内存文档记录
# 注意：服务重启后此字典会清空
# Chroma 中的向量数据不会清空
# =========================================================

DOCUMENT_STORE: dict[str, dict[str, Any]] = {}


# =========================================================
# Embedding 与 Chroma
# =========================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

vector_collection = (
    chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME
    )
)


# =========================================================
# 请求模型
# =========================================================

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
    )


class AskDocumentRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
    )

    doc_id: str | None = None

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )


class VectorSearchDocumentRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
    )

    doc_id: str | None = None

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )


class AskBlogRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
    )

    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
    )


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",
        "http://127.0.0.1:4321",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 基础接口
# =========================================================

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
            "POST /sync-blog",
            "POST /ask-blog",
        ],
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "mini-rag-assistant",
        "memory_document_count": len(
            DOCUMENT_STORE
        ),
        "vector_count": vector_collection.count(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "deepseek_model": DEEPSEEK_MODEL,
    }


# =========================================================
# 普通聊天
# =========================================================

@app.post("/chat")
def chat(request: ChatRequest):
    check_api_key()

    try:
        completion = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. "
                        "Explain things clearly and simply."
                    ),
                },
                {
                    "role": "user",
                    "content": request.message,
                },
            ],
            stream=False,
        )

        return {
            "reply": (
                completion
                .choices[0]
                .message
                .content
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat service error: {exc}",
        ) from exc


# =========================================================
# 上传 TXT / Markdown
# =========================================================

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...)
):
    allowed_extensions = {
        ".txt",
        ".md",
    }

    original_filename = Path(
        file.filename or "document.txt"
    ).name

    file_suffix = Path(
        original_filename
    ).suffix.lower()

    if file_suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only .txt and .md files "
                "are supported for now."
            ),
        )

    doc_id = str(uuid.uuid4())
    saved_filename = f"{doc_id}{file_suffix}"
    file_path = UPLOAD_DIR / saved_filename

    try:
        content_bytes = await file.read()

        try:
            text = content_bytes.decode("utf-8")

        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="File must be encoded in UTF-8.",
            ) from exc

        chunks = chunk_text(text)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        file_path.write_text(
            text,
            encoding="utf-8",
        )

        DOCUMENT_STORE[doc_id] = {
            "filename": original_filename,
            "saved_path": str(file_path),
            "chunk_count": len(chunks),
        }

        chunk_ids = [
            f"{doc_id}_{index}"
            for index in range(len(chunks))
        ]

        chunk_metadatas = [
            {
                "source_type": "uploaded_document",
                "doc_id": doc_id,
                "filename": original_filename,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        chunk_embeddings = (
            embedding_model
            .encode(chunks)
            .tolist()
        )

        vector_collection.upsert(
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

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {exc}",
        ) from exc


# =========================================================
# 普通文档向量搜索
# =========================================================

@app.post("/vector-search-document")
def vector_search_document(
    request: VectorSearchDocumentRequest
):
    if vector_collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "No document embeddings found. "
                "Please upload or sync content first."
            ),
        )

    results = vector_search(
        query=request.query,
        top_k=request.top_k,
        doc_id=request.doc_id,
        source_type=(
            "uploaded_document"
            if not request.doc_id
            else None
        ),
    )

    return {
        "query": request.query,
        "top_k": request.top_k,
        "result_count": len(results),
        "results": results,
    }


# =========================================================
# 普通上传文档问答
# =========================================================

@app.post("/ask-document-vector")
def ask_document_vector(
    request: AskDocumentRequest
):
    check_api_key()

    if vector_collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "No document embeddings found. "
                "Please upload a document first."
            ),
        )

    top_results = vector_search(
        query=request.question,
        top_k=request.top_k,
        doc_id=request.doc_id,
        source_type=(
            None
            if request.doc_id
            else "uploaded_document"
        ),
    )

    if not top_results:
        return {
            "question": request.question,
            "answer": (
                "I don't know based on "
                "the uploaded documents."
            ),
            "citations": [],
            "retrieved_chunks": [],
        }

    context = "\n\n".join(
        (
            f"[Source {index}]\n"
            f"Filename: {result.get('filename')}\n"
            f"Chunk index: "
            f"{result.get('chunk_index')}\n"
            f"Content:\n{result['text']}"
        )
        for index, result in enumerate(
            top_results,
            start=1,
        )
    )

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question using only the provided context.

Rules:
- If the answer is not in the context, say:
  "I don't know based on the uploaded documents."
- Do not use outside knowledge.
- Keep the answer clear and concise.
- Include citation markers like [Source 1].
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
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            stream=False,
        )

        return {
            "question": request.question,
            "answer": (
                response
                .choices[0]
                .message
                .content
            ),
            "retrieval_method": "vector_search",
            "citations": [
                {
                    "source": f"Source {index}",
                    "doc_id": result.get("doc_id"),
                    "filename": result.get(
                        "filename"
                    ),
                    "chunk_index": result.get(
                        "chunk_index"
                    ),
                    "distance": result.get(
                        "distance"
                    ),
                }
                for index, result in enumerate(
                    top_results,
                    start=1,
                )
            ],
            "retrieved_chunks": top_results,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Vector document QA failed: {exc}",
        ) from exc


# =========================================================
# Astro 博客同步
# =========================================================

@app.post("/sync-blog")
def sync_blog():
    try:
        result = sync_all_blog_posts(
            vector_collection=vector_collection,
            embedding_model=embedding_model,
            blog_base_url=BLOG_BASE_URL,
            blog_url_prefix=BLOG_URL_PREFIX,
        )

        return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Blog synchronization failed: {exc}",
        ) from exc


# =========================================================
# Astro 博客问答
# =========================================================

@app.post("/ask-blog")
def ask_blog(
    request: AskBlogRequest
):
    check_api_key()

    if vector_collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "No vector data found. "
                "Please call POST /sync-blog first."
            ),
        )

    top_results = vector_search(
        query=request.question,
        top_k=request.top_k,
        source_type="astro_blog",
    )

    # 这里控制“博客是否足够相关”
    # distance 越小通常代表越相似
    BLOG_RELEVANCE_THRESHOLD = 0.95

    relevant_results = [
        result
        for result in top_results
        if result.get("distance") is not None
        and result["distance"] <= BLOG_RELEVANCE_THRESHOLD
    ]

    # =====================================================
    # 情况 1：博客中找到足够相关内容
    # =====================================================

    if relevant_results:
        context = "\n\n".join(
            (
                f"[Source {index}]\n"
                f"文章标题："
                f"{result.get('title') or ''}\n"
                f"章节："
                f"{result.get('heading') or ''}\n"
                f"链接："
                f"{result.get('url') or ''}\n"
                f"Chunk："
                f"{result.get('chunk_index')}\n"
                f"博客内容：\n"
                f"{result['text']}"
            )
            for index, result in enumerate(
                relevant_results,
                start=1,
            )
        )

        system_prompt = """
你是这个网站的个人博客知识助手。

当前已经检索到与用户问题相关的博客内容。

请优先并严格根据博客上下文回答。

规则：
1. 所有关键事实都必须来自博客上下文。
2. 不要使用博客之外的知识补充博客观点。
3. 使用中文回答。
4. 使用 [Source 1]、[Source 2] 这样的标记引用来源。
5. 不要编造文章标题、URL、章节或引用。
6. 回答要清晰、准确。
"""

        user_prompt = f"""
用户问题：
{request.question}

检索到的博客上下文：
{context}
"""

        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                stream=False,
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

            citations = [
                {
                    "source": f"Source {index}",
                    "title": result.get("title"),
                    "slug": result.get("slug"),
                    "url": result.get("url"),
                    "heading": result.get("heading"),
                    "filename": result.get("filename"),
                    "chunk_index": result.get(
                        "chunk_index"
                    ),
                    "distance": result.get(
                        "distance"
                    ),
                }
                for index, result in enumerate(
                    relevant_results,
                    start=1,
                )
            ]

            return {
                "question": request.question,
                "answer": answer,
                "answer_mode": "blog",
                "retrieval_method": (
                    "astro_blog_vector_search"
                ),
                "citations": citations,
                "retrieved_chunks": relevant_results,
            }

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Blog QA failed: {exc}",
            ) from exc

    # =====================================================
    # 情况 2：博客中没有足够相关内容
    # 交给外部 LLM 使用通用知识回答
    # =====================================================

    general_system_prompt = """
你是一个通用 AI 助手。

当前博客知识库没有找到足够相关的内容，
请使用你的通用知识回答用户问题。

规则：
1. 使用中文回答。
2. 明确说明本次回答不是来自博客内容。
3. 不要伪造博客引用。
4. 不要声称答案来自用户的博客。
5. 回答要清晰、准确。
"""

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": general_system_prompt,
                },
                {
                    "role": "user",
                    "content": request.question,
                },
            ],
            stream=False,
        )

        general_answer = (
            response
            .choices[0]
            .message
            .content
        )

        return {
            "question": request.question,
            "answer": (
                "以下回答来自通用 AI 知识，"
                "并非来自我的博客：\n\n"
                f"{general_answer}"
            ),
            "answer_mode": "general",
            "retrieval_method": "external_llm",
            "citations": [],
            "retrieved_chunks": [],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"General AI response failed: {exc}",
        ) from exc

# =========================================================
# 公共函数
# =========================================================

def check_api_key() -> None:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "DEEPSEEK_API_KEY is missing. "
                "Please set it in your .env file."
            ),
        )


def vector_search(
    query: str,
    top_k: int = 3,
    doc_id: str | None = None,
    source_type: str | None = None,
    distance_threshold: float = (
        VECTOR_DISTANCE_THRESHOLD
    ),
) -> list[dict[str, Any]]:
    """
    执行 Chroma 向量搜索。

    source_type:
        uploaded_document
        astro_blog
    """
    query_embedding = (
        embedding_model
        .encode([query])
        .tolist()
    )

    filters: list[dict[str, Any]] = []

    if doc_id:
        filters.append({
            "doc_id": doc_id
        })

    if source_type:
        filters.append({
            "source_type": source_type
        })

    where_filter: dict[str, Any] | None

    if len(filters) == 0:
        where_filter = None

    elif len(filters) == 1:
        where_filter = filters[0]

    else:
        where_filter = {
            "$and": filters
        }

    query_arguments: dict[str, Any] = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where_filter is not None:
        query_arguments["where"] = where_filter

    try:
        results = vector_collection.query(
            **query_arguments
        )

    except Exception as exc:
        raise RuntimeError(
            f"Vector search failed: {exc}"
        ) from exc

    documents = (
        results.get("documents")
        or [[]]
    )[0]

    metadatas = (
        results.get("metadatas")
        or [[]]
    )[0]

    distances = (
        results.get("distances")
        or [[]]
    )[0]

    search_results: list[dict[str, Any]] = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        if document is None:
            continue

        if metadata is None:
            metadata = {}

        numeric_distance = float(distance)

        if numeric_distance > distance_threshold:
            continue

        search_results.append({
            "source_type": metadata.get(
                "source_type"
            ),
            "doc_id": metadata.get("doc_id"),
            "filename": metadata.get(
                "filename"
            ),
            "slug": metadata.get("slug"),
            "title": metadata.get("title"),
            "url": metadata.get("url"),
            "heading": metadata.get(
                "heading"
            ),
            "chunk_index": metadata.get(
                "chunk_index"
            ),
            "distance": numeric_distance,
            "text": document,
        })

    return search_results


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    if not isinstance(text, str):
        raise TypeError(
            "text must be a string"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 "
            "and smaller than chunk_size"
        )

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks
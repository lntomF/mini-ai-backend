# AI / GenAI / RAG / Agent 实习冲刺学习上下文

## 1. 我的身份和目标

我是一个 AI / GenAI / RAG / Agent 方向的小白学习者，目前目标是：

> 在暑假尽快具备申请 AI Engineer Intern、GenAI Intern、RAG Intern、LLM Application Intern、AI Agent Intern 的能力。

我不是要泛泛学习概念，而是想快速做出可以放进 GitHub、简历、面试项目介绍里的作品集项目。

我希望后续模型回答时遵循：

- 用中文回答
- 像实习冲刺教练一样直接、具体、可执行
- 不要一上来讲太复杂的架构
- 不要假设我是高级工程师
- 每一步都要能转化成代码、README、GitHub、简历 bullet、面试话术
- 我现在最需要的是：项目、简历、面试表达、投递准备

---

## 2. 我的当前真实水平

我目前是小白阶段。

我已经能：

- 跑通 FastAPI 项目
- 接入 DeepSeek API
- 理解基础 LLM API 调用
- 理解 structured output 的基本作用
- 理解 tool calling 的基本流程
- 理解 RAG 的基础流程：
  - 上传文档
  - 文本切块
  - 检索相关 chunk
  - 把 chunk 放进 prompt
  - LLM 基于文档回答
  - 不知道就拒答
  - 返回 citation

但我还不是完全独立写出来的。很多代码是跟着教练一步步 copy、修改、测试完成的。

我当前阶段是：

> 已经从 copy 代码进入到能解释、能调试、能小幅修改项目的阶段。

后续模型不要直接把我推到 LangGraph、多 Agent、MCP、复杂 Tool Registry、生产级部署等高级内容，除非我明确要求。

---

## 3. 当前主项目名称

项目名称：

```text
Mini RAG Assistant with DeepSeek
```

项目定位：

> 一个基于 FastAPI + DeepSeek API + Sentence Transformers + ChromaDB 的小型 RAG Assistant，支持文档上传、chunking、关键词检索、向量检索、基于文档问答、citation、拒答和基础评估。

---

## 4. 当前技术栈

目前项目用到：

```text
Python
FastAPI
DeepSeek API
OpenAI-compatible SDK
Pydantic
Uvicorn
python-dotenv
python-multipart
Sentence Transformers
ChromaDB
```

本地环境中有：

```text
.env
requirements.txt
main.py
uploads/
chroma_db/
```

`.env` 里配置：

```env
DEEPSEEK_API_KEY=我的真实 key
DEEPSEEK_MODEL=deepseek-v4-flash
```

注意：真实 API key 不应该提交到 GitHub。

`.gitignore` 应该包含：

```gitignore
.env
.venv/
__pycache__/
*.pyc
.DS_Store
uploads/
chroma_db/
```

---

## 5. 已完成的 API 功能

### 5.1 健康检查 `/`

已完成。

作用：

```text
确认 FastAPI 服务正在运行，并返回当前支持的 endpoints。
```

---

### 5.2 本地文本分析 `/analyze`

已完成。

请求示例：

```json
{
  "text": "I want to become an AI engineer intern"
}
```

作用：

```text
不调用 AI，只在本地统计 word_count、char_count。
```

这是用来练习 FastAPI request / response 的基础接口。

---

### 5.3 基础聊天 `/chat`

已完成。

请求示例：

```json
{
  "message": "什么是 RAG？"
}
```

作用：

```text
后端接收用户 message，调用 DeepSeek API，返回普通 LLM 聊天回答。
```

我对它的理解：

```text
/chat 是最基础的 LLM API 调用接口。
用户输入 message，FastAPI 后端把 message 发给 DeepSeek，然后把模型回答返回给用户。
```

---

### 5.4 意图分类 `/classify`

已完成。

请求示例：

```json
{
  "message": "帮我 debug 这段 Python 代码"
}
```

返回示例：

```json
{
  "intent": "coding",
  "confidence": 0.9,
  "reason": "The user asks for help with code debugging."
}
```

作用：

```text
通过 structured output 约束模型返回标准 JSON，用来判断用户意图。
```

目前支持的意图包括：

```text
qa
summarization
translation
coding
writing
calculation
text_analysis
unknown
```

我对它的理解：

```text
/classify 不是让模型随便聊天，而是让模型按固定 JSON 格式输出，方便后端程序继续处理。
```

---

### 5.5 工具调用 Agent `/agent`

已完成。

请求示例：

```json
{
  "message": "帮我计算 23 * 19 + 7"
}
```

支持工具：

```text
calculator
word_counter
```

作用：

```text
模型先判断是否需要调用工具。
如果不需要工具，直接回答。
如果需要工具，模型返回 tool_calls。
后端解析 tool_calls，执行对应 Python 函数。
然后后端把工具结果发回模型，模型基于工具结果生成最终回答。
```

我对它的理解：

```text
LLM 负责理解用户意图和选择工具，Python 工具负责准确执行。
比如数学计算不完全依赖大模型自己猜，而是交给 calculator 工具，减少幻觉。
```

---

### 5.6 学习计划生成 `/study-plan`

已指导加入。

请求示例：

```json
{
  "goal": "AI Agent 实习",
  "days": 7
}
```

作用：

```text
调用 DeepSeek，根据目标和天数生成学习计划。
```

这个接口主要是让我练习：

```text
如何新增 FastAPI endpoint
如何定义 Pydantic Request Model
如何做参数校验
如何把用户输入拼进 prompt
如何调用 LLM 返回结果
```

---

## 6. RAG 项目进度

当前已经完成 Mini RAG Assistant 的核心流程。

---

### 6.1 文档上传 `/upload-document`

已完成。

支持文件类型：

```text
.txt
.md
```

上传后做的事情：

```text
1. 检查文件扩展名
2. 读取 UTF-8 文本
3. 保存到 uploads/
4. 生成 doc_id
5. 调用 chunk_text 切分文档
6. 把 chunks 保存到 DOCUMENT_STORE
7. 生成 embeddings
8. 存入 ChromaDB
9. 返回 char_count、chunk_count、chunks_preview
```

返回示例结构：

```json
{
  "doc_id": "xxxx",
  "filename": "test.txt",
  "saved_path": "uploads/xxxx_test.txt",
  "char_count": 300,
  "chunk_count": 1,
  "chunks_preview": [
    "RAG stands for Retrieval-Augmented Generation..."
  ]
}
```

---

### 6.2 文本切块 `chunk_text`

已完成。

当前 chunking 策略：

```text
character-based chunking
```

配置：

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
```

理解：

```text
每 500 个字符切一段，相邻 chunk 重叠 80 个字符，避免上下文被硬切断。
```

目前够用，但未来可以升级为：

```text
按段落切
按句子切
按 token 切
保留 metadata
```

---

### 6.3 内存文档库 `DOCUMENT_STORE`

已完成。

形式：

```python
DOCUMENT_STORE = {}
```

保存结构：

```python
DOCUMENT_STORE[doc_id] = {
    "filename": file.filename,
    "saved_path": str(file_path),
    "text": text,
    "chunks": chunks
}
```

注意：

```text
DOCUMENT_STORE 是内存存储，服务重启后会清空。
ChromaDB 是本地持久化的，但 DOCUMENT_STORE 不是。
```

这是当前项目的一个 limitation。

---

### 6.4 关键词检索 `/search-document`

已完成。

请求示例：

```json
{
  "query": "hallucination context",
  "top_k": 3
}
```

返回示例：

```json
{
  "query": "hallucination context",
  "top_k": 3,
  "result_count": 1,
  "results": [
    {
      "doc_id": "25589b03-3ca7-45c8-87cb-f57c4e71e3b4",
      "filename": "新建 文本文档 (2).txt",
      "chunk_index": 0,
      "score": 2,
      "text": "RAG stands for Retrieval-Augmented Generation..."
    }
  ]
}
```

---

### 6.5 Stopwords 过滤

已完成。

曾经遇到的问题：

```text
用户问：Who is the CEO of NVIDIA?
文档里没有 NVIDIA CEO。
但关键词检索因为匹配了 is、the、who、of 等普通词，错误返回了 RAG 文档 chunk。
```

解决方案：

```text
增加 STOPWORDS，过滤无意义高频词。
```

还增加了：

```text
matched_words
```

用来 debug 为什么检索命中了某个 chunk。

示例：

```json
"matched_words": [
  "rag",
  "reduce",
  "hallucination"
]
```

---

### 6.6 基于关键词的文档问答 `/ask-document`

已完成。

流程：

```text
用户提问
↓
keyword_search 找相关 chunks
↓
把 chunks 拼成 context
↓
DeepSeek 只基于 context 回答
↓
返回 answer、citations、retrieved_chunks
```

成功例子：

请求：

```json
{
  "question": "How does RAG reduce hallucination?",
  "top_k": 3
}
```

返回：

```json
{
  "question": "How does RAG reduce hallucination?",
  "answer": "RAG reduces hallucination because answers are grounded in retrieved context [Source 1].",
  "citations": [
    {
      "source": "Source 1",
      "filename": "新建 文本文档 (2).txt",
      "chunk_index": 0
    }
  ]
}
```

拒答例子：

请求：

```json
{
  "question": "Who is the CEO of NVIDIA?",
  "top_k": 3
}
```

理想返回：

```json
{
  "answer": "I don't know based on the uploaded documents.",
  "citations": [],
  "retrieved_chunks": []
}
```

---

### 6.7 Embedding + ChromaDB 向量检索 `/vector-search-document`

已完成。

使用：

```text
Sentence Transformers
ChromaDB
```

大致配置：

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "rag_documents"
```

流程：

```text
上传文档后：
chunks → embeddings → ChromaDB

用户搜索时：
query → embedding → ChromaDB 相似度搜索 → 返回相似 chunks
```

成功测试：

请求：

```json
{
  "query": "How can RAG avoid making things up?",
  "top_k": 3
}
```

返回：

```json
{
  "query": "How can RAG avoid making things up?",
  "top_k": 3,
  "result_count": 1,
  "results": [
    {
      "doc_id": "2a04ebf4-ec2d-4058-9b1b-058803952146",
      "filename": "新建 文本文档 (2).txt",
      "chunk_index": 0,
      "distance": 0.934402585029602,
      "text": "RAG stands for Retrieval-Augmented Generation..."
    }
  ]
}
```

理解：

```text
向量检索不是只看关键词，而是看语义相似度。
比如用户问 avoid making things up，文档中说 reduce hallucination / grounded in retrieved context，vector search 能找到相关 chunk。
```

---

### 6.8 向量距离阈值 `VECTOR_DISTANCE_THRESHOLD`

已完成或已指导完成。

遇到的问题：

```text
向量搜索默认总会返回最相近的 top_k 个结果，即使结果其实不相关。
```

例子：

请求：

```json
{
  "query": "Who is the CEO of NVIDIA?",
  "top_k": 3
}
```

曾经返回：

```json
{
  "distance": 2.314012289047241,
  "text": "RAG stands for Retrieval-Augmented Generation..."
}
```

这是不相关结果。

解决方案：

```python
VECTOR_DISTANCE_THRESHOLD = 1.5
```

规则：

```text
distance <= 1.5：认为相关
distance > 1.5：过滤掉
```

已观察到：

```text
相关问题 distance 大约 0.93
不相关问题 distance 大约 2.31
```

所以 1.5 是当前测试集下的合理临时阈值。

---

### 6.9 向量版文档问答 `/ask-document-vector`

已指导实现。

目标流程：

```text
用户提问
↓
问题转 embedding
↓
ChromaDB 找语义相关 chunks
↓
过滤 distance 太大的 chunk
↓
把 chunks 放进 prompt
↓
DeepSeek 只基于 chunks 回答
↓
返回 answer、retrieval_method、citations、retrieved_chunks
```

请求格式应该是：

```json
{
  "question": "How can RAG avoid making things up?",
  "top_k": 3
}
```

注意：

如果接口绑定的是 `VectorSearchDocumentRequest`，就会要求字段 `query`。

如果接口绑定的是 `AskDocumentRequest`，就会要求字段 `question`。

曾经遇到过 422：

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "query"],
      "msg": "Field required"
    }
  ]
}
```

原因：

```text
测试 /vector-search-document 时用了 question 字段；
该接口实际需要 query 字段。
```

要区分：

```text
/vector-search-document 用 query
/ask-document-vector 用 question
```

---

### 6.10 RAG 评估 `/evaluate-rag`

已完成，并通过测试。

作用：

```text
自动跑一组小测试，检查 RAG 是否能：
1. 回答文档内问题
2. 回答语义改写问题
3. 拒答文档外问题
4. 返回 citation
5. 不乱引用
```

当前评估用例：

```python
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
```

实际返回：

```json
{
  "total_cases": 3,
  "passed_cases": 3,
  "pass_rate": 1,
  "results": [
    {
      "question": "How does RAG reduce hallucination?",
      "should_answer": true,
      "answer": "RAG reduces hallucination because answers are grounded in retrieved context [Source 1].",
      "citations": [
        {
          "source": "Source 1",
          "doc_id": "2a04ebf4-ec2d-4058-9b1b-058803952146",
          "filename": "新建 文本文档 (2).txt",
          "chunk_index": 0,
          "distance": 1.1127302646636963
        }
      ],
      "passed": true,
      "reason": "Answer contains expected keywords."
    },
    {
      "question": "How can RAG avoid making things up?",
      "should_answer": true,
      "answer": "RAG avoids making things up by grounding answers in retrieved context, which reduces hallucination [Source 1].",
      "citations": [
        {
          "source": "Source 1",
          "doc_id": "2a04ebf4-ec2d-4058-9b1b-058803952146",
          "filename": "新建 文本文档 (2).txt",
          "chunk_index": 0,
          "distance": 0.934402585029602
        }
      ],
      "passed": true,
      "reason": "Answer contains expected keywords."
    },
    {
      "question": "Who is the CEO of NVIDIA?",
      "should_answer": false,
      "answer": "I don't know based on the uploaded documents.",
      "citations": [],
      "passed": true,
      "reason": "Expected refusal answer."
    }
  ]
}
```

结论：

```text
/evaluate-rag 已通过，Mini RAG Assistant v1 可以阶段性收尾。
```

---

## 7. 我已经学到的核心概念

我已经接触并初步理解：

```text
FastAPI endpoint
Pydantic request validation
422 Unprocessable Entity
.env 环境变量
DeepSeek API 调用
OpenAI-compatible SDK
structured output
tool calling
function calling
RAG
chunking
keyword search
stopwords
matched_words debug
embedding
vector search
ChromaDB
distance threshold
citation metadata
refusal behavior
RAG evaluation
```

---

## 8. 曾经遇到过的重要 bug / 问题

### 8.1 422 Unprocessable Entity

原因通常是请求 body 字段名不匹配。

例子：

`/chat` 需要：

```json
{
  "message": "..."
}
```

不能传：

```json
{
  "text": "..."
}
```

`/analyze` 才用：

```json
{
  "text": "..."
}
```

`/vector-search-document` 用：

```json
{
  "query": "..."
}
```

`/ask-document-vector` 用：

```json
{
  "question": "..."
}
```

---

### 8.2 关键词检索误召回

问题：

```text
普通词 is / the / who / of / can 等导致不相关 chunk 被召回。
```

解决：

```text
加 STOPWORDS
返回 matched_words 方便 debug
```

---

### 8.3 向量检索总会返回结果

问题：

```text
ChromaDB top_k 会返回最相近的结果，即使不相关。
```

解决：

```text
加 VECTOR_DISTANCE_THRESHOLD
distance 太大就过滤
```

---

## 9. 当前项目 README 应该怎么写

建议 README 标题：

```md
# Mini RAG Assistant with DeepSeek
```

项目介绍：

```md
A FastAPI-based RAG assistant that supports document upload, chunking, semantic vector search, grounded question answering, citation metadata, refusal behavior, and basic RAG evaluation.
```

Features：

```md
## Features

- Upload `.txt` and `.md` documents
- Split documents into overlapping chunks
- Store document chunks in memory
- Generate embeddings with Sentence Transformers
- Store and search vectors with ChromaDB
- Support keyword-based retrieval
- Support vector-based semantic retrieval
- Answer questions based only on retrieved document context
- Return citation metadata
- Refuse to answer out-of-context questions
- Evaluate the RAG pipeline with a small test set
```

System Flow：

```md
## System Flow

```text
User uploads document
        ↓
Backend reads text
        ↓
Text is split into chunks
        ↓
Chunks are embedded with Sentence Transformers
        ↓
Embeddings are stored in ChromaDB
        ↓
User asks a question
        ↓
Question is embedded
        ↓
ChromaDB retrieves similar chunks
        ↓
DeepSeek answers using only retrieved context
        ↓
API returns answer, citations, and retrieved chunks
```

```

Main APIs：

```md
## Main APIs

### Chat API

POST /chat

Basic LLM chat endpoint using DeepSeek API.

### Document Upload

POST /upload-document

Uploads a `.txt` or `.md` document, saves it locally, splits it into chunks, and stores embeddings in ChromaDB.

### Keyword Search

POST /search-document

Searches document chunks using keyword matching and stopword filtering.

### Vector Search

POST /vector-search-document

Searches document chunks using embedding-based semantic retrieval.

### Vector-based RAG QA

POST /ask-document-vector

Retrieves relevant chunks with vector search and asks DeepSeek to answer only based on the retrieved context.

### RAG Evaluation

POST /evaluate-rag

Runs a small evaluation set to check answer correctness, citation behavior, and refusal behavior.
```

---

## 10. 当前项目的限制

目前项目还不完善，limitations 包括：

```text
1. 只支持 .txt 和 .md 文件
2. 暂时没有 PDF parsing
3. DOCUMENT_STORE 是内存存储，服务重启会丢
4. ChromaDB 本地存储，不是云端数据库
5. 没有前端 UI
6. 没有登录和权限控制
7. 没有 Docker
8. 没有部署上线
9. evaluation 只有 3 个简单测试样例
10. chunking 还是 character-based，不是 token-based
11. 没有 reranker
12. 没有 hybrid retrieval
13. 没有 tracing / logging
```

---

## 11. 当前可以写进简历的 bullet

较完整版本：

```text
Built a FastAPI-based Mini RAG Assistant integrating DeepSeek API, Sentence Transformers, and ChromaDB, supporting document upload, chunking, semantic vector search, grounded question answering, citation metadata, refusal handling, and a basic RAG evaluation endpoint.
```

更短版本：

```text
Developed a Mini RAG Assistant with FastAPI, DeepSeek API, Sentence Transformers, and ChromaDB, enabling document-based QA with vector retrieval, citations, refusal behavior, and evaluation.
```

部署后可以升级成：

```text
Built and deployed a FastAPI-based Mini RAG Assistant with DeepSeek API, Sentence Transformers, and ChromaDB, supporting document upload, vector retrieval, grounded QA, citations, refusal behavior, evaluation, and a blog-embedded demo interface.
```

---

## 12. 面试项目介绍话术

我可以这样介绍这个项目：

```text
我做了一个 Mini RAG Assistant。用户可以上传 txt 或 md 文档，后端会读取文本并切成 overlapping chunks，然后用 Sentence Transformers 生成 embeddings，存进 ChromaDB。

用户提问时，系统会把问题也转成 embedding，然后在 ChromaDB 里做语义检索，找出最相关的 chunks。接着我把这些 chunks 作为 context 传给 DeepSeek，并要求模型只能基于 context 回答。

如果检索不到足够相关的内容，系统会拒答，返回 “I don't know based on the uploaded documents.”，避免模型用外部知识乱答。

我还实现了 citation metadata，返回每个回答引用的是哪个 chunk。另外我做了一个简单的 /evaluate-rag 接口，用几个测试问题检查系统是否能回答文档内问题、语义改写问题，以及拒答文档外问题。
```

如果被问到关键词检索和向量检索区别，可以这样答：

```text
我实现了两种检索方式：keyword-based retrieval 和 embedding-based vector retrieval。关键词检索简单直观，但是只能匹配字面词，容易受表达方式影响；向量检索会把 query 和 document chunks 转成 embeddings，通过语义相似度找相关内容，所以即使用户表达和原文不完全一样，也能找到相关 chunk。
```

如果被问到为什么需要 distance threshold：

```text
因为向量数据库 top_k 默认总会返回最相近的结果，即使结果其实不相关。所以我加了 distance threshold，只有距离低于阈值的 chunk 才会被交给模型，否则系统拒答。这样可以减少错误引用和不相关上下文导致的幻觉。
```

---

## 13. 下一步目标：部署到我的 Blog 上

我现在想把这个项目部署到我的 blog 上，作为作品集展示。

正确部署结构应该是：

```text
我的 Blog 页面 = 前端展示入口
FastAPI RAG 项目 = 后端 API 服务
```

架构：

```text
User
 ↓
Blog Demo Page
 ↓ fetch()
FastAPI API on Render/Railway/VPS
 ↓
DeepSeek API
 ↓
ChromaDB / uploaded documents
 ↓
返回 answer + citations
```

推荐小白部署方案：

```text
FastAPI 后端：Render 或 Railway
Blog 前端：HTML/JS demo 页面
```

需要做的事情：

```text
1. main.py 加 CORS
2. main.py 加 /health
3. requirements.txt 确认完整
4. .gitignore 确认没有提交 .env、uploads/、chroma_db/
5. 部署 FastAPI 到 Render 或 Railway
6. 获取公网 API URL
7. 在 blog 页面放 HTML/JS demo
8. Demo 调用 /upload-document 和 /ask-document-vector
```

CORS 示例：

```python
from fastapi.middleware.cors import CORSMiddleware

allowed_origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://我的博客域名.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

健康检查接口：

```python
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "mini-rag-assistant"
    }
```

Render/Railway 启动命令：

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

环境变量：

```text
DEEPSEEK_API_KEY
DEEPSEEK_MODEL
```

注意：

```text
sentence-transformers + chromadb 可能比较重，免费部署平台可能启动慢或内存不够。
如果失败，下一步可以考虑 Docker 部署，或者把 embedding 换成云端 embedding API。
```

---

## 14. 我希望其他模型接下来怎么帮我

请基于以上上下文继续带我，不要重新从 0 开始。

我最希望下一步帮我做：

```text
1. 帮我把项目部署到我的 blog 上
2. 先问清楚我的 blog 类型：GitHub Pages / Vercel / WordPress / 自己服务器
3. 给我最小可行部署步骤
4. 不要一上来搞复杂 DevOps
5. 如果部署失败，帮我逐个 debug
6. 帮我最终整理成 GitHub README、博客文章、简历 bullet、面试讲法
```

当前最优下一步：

```text
确认我的 blog 平台，然后指导我：
- FastAPI 加 CORS 和 /health
- 部署后端
- 写一个最小 HTML demo
- 嵌入 blog
- 测试上传文档和问答
```

---

## 15. 给后续模型的注意事项

请记住：

```text
我不是高级工程师。
我目标是暑假找 AI / GenAI / RAG / Agent 实习。
我需要项目导向、结果导向、简历导向。
我已经有一个本地可运行的 Mini RAG Assistant。
现在最重要的是部署、展示、简历和面试表达。
```

不要优先推荐：

```text
LangGraph
MCP
复杂多 Agent
复杂 Tool Registry
Kubernetes
生产级微服务
复杂云架构
```

除非我明确要求。

请优先帮助我完成：

```text
GitHub 项目可展示
Blog demo 可访问
README 专业
简历 bullet 有工程味
面试能讲清楚
```

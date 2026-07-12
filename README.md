# 迷你 AI RAG 后端

这是一个基于 **FastAPI + DeepSeek + ChromaDB** 的轻量级 RAG 系统。支持两大功能模块：

1. **文档 RAG**：上传文档、向量检索、基于文档的问答
2. **博客同步与问答**：自动同步 Astro 博客、向量化存储、基于博客内容的问答

## 实现效果

![alt text](<image/屏幕截图 2026-07-11 212834.png>)
![alt text](<image/屏幕截图 2026-07-11 212839.png>)
![alt text](<image/屏幕截图 2026-07-11 212852.png>)

## 环境要求

- Python 3.10+
- 可访问 DeepSeek API 的网络环境
- 首次运行会自动创建 `uploads/` 和 `chroma_db/`

## 安装

```bash
pip install -r requirements.txt
```

## 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# 必填：DeepSeek API Key
DEEPSEEK_API_KEY=你的key

# 可选：DeepSeek 模型，不配置时默认使用 deepseek-chat
DEEPSEEK_MODEL=deepseek-chat

# 可选：Astro 博客配置（如果要使用博客同步功能）
BLOG_BASE_URL=http://localhost:4321
BLOG_URL_PREFIX=/blog
```

说明：

- `DEEPSEEK_API_KEY` **必填**，否则 `/chat` 和问答接口会报错
- `DEEPSEEK_MODEL` 可选，默认值为 `deepseek-chat`
- `BLOG_BASE_URL` 和 `BLOG_URL_PREFIX` 用于配置博客源，如果不使用博客功能可以不配置

## 启动服务

```bash
uvicorn main:app --reload
```

启动后默认可访问：

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

## 接口速览

### 基础接口

#### 1. `GET /`

返回服务状态和当前开放接口列表。

示例响应：

```json
{
   "message": "Mini RAG Assistant is running",
   "endpoints": [
      "GET /",
      "GET /health",
      "POST /chat",
      "POST /upload-document",
      "POST /vector-search-document",
      "POST /ask-document-vector",
      "POST /sync-blog",
      "POST /ask-blog"
   ]
}
```

#### 2. `GET /health`

返回健康检查信息，包括内存中的文档数、向量库中的总 chunk 数、正在使用的模型。

示例响应：

```json
{
   "status": "ok",
   "service": "mini-rag-assistant",
   "memory_document_count": 1,
   "vector_count": 8,
   "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
   "deepseek_model": "deepseek-chat"
}
```

#### 3. `POST /chat`

直接调用 DeepSeek 做普通对话。

请求体：

```json
{
   "message": "你好，帮我介绍一下这个项目"
}
```

返回示例：

```json
{
   "reply": "..."
}
```

### 文档 RAG 接口

#### 4. `POST /upload-document`

上传 `.txt` 或 `.md` 文件，服务会：

- 校验文件扩展名
- 按 UTF-8 读取文本
- 按固定长度切分 chunk
- 保存原文件到 `uploads/`
- 生成 embedding 并写入 ChromaDB

请求方式：`multipart/form-data`

返回示例：

```json
{
   "doc_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
   "filename": "notes.md",
   "char_count": 1234,
   "chunk_count": 6,
   "chunks_preview": ["...", "...", "..."]
}
```

#### 5. `POST /vector-search-document`

对已上传文档执行向量语义检索。

请求体：

```json
{
   "query": "什么是人工智能？",
   "doc_id": null,
   "top_k": 3
}
```

说明：

- `doc_id` 为空时，会在所有已上传文档中检索
- `top_k` 范围是 1 到 10
- 代码里使用了距离阈值过滤，过远的结果会被丢弃

返回示例：

```json
{
   "query": "什么是人工智能？",
   "top_k": 3,
   "result_count": 2,
   "results": [
      {
         "doc_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
         "filename": "notes.md",
         "chunk_index": 1,
         "distance": 0.42,
         "text": "..."
      }
   ]
}
```

#### 6. `POST /ask-document-vector`

先做向量召回，再把召回到的内容交给 DeepSeek 回答问题。

请求体：

```json
{
   "question": "文档里对 FastAPI 的描述是什么？",
   "doc_id": null,
   "top_k": 3
}
```

返回示例：

```json
{
   "question": "文档里对 FastAPI 的描述是什么？",
   "answer": "...",
   "retrieval_method": "vector_search",
   "citations": [
      {
         "source": "Source 1",
         "doc_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
         "filename": "notes.md",
         "chunk_index": 0,
         "distance": 0.39
      }
   ],
   "retrieved_chunks": []
}
```

如果没有检索到内容，接口会直接返回：

```json
{
   "answer": "I don't know based on the uploaded documents."
}
```

### 博客同步与问答接口

#### 7. `POST /sync-blog`

同步 Astro 博客文章到向量库。将博客中的所有 markdown 文件进行：

- 内容加载
- Markdown 解析（提取标题和正文）
- 文本切块
- Embedding 生成
- 向量化存储

当前实现会先完成 embedding 和写入，再清理过期旧 chunk，避免出现“旧数据已删除但新数据写入失败”的丢失风险。

另外，索引时 `chunks` 和 `raw_texts` 是刻意分开的：

- `chunks`：用于 embedding、稳定 ID 生成和检索语义输入
- `raw_texts`：用于写入 Chroma 的 `documents`，保留原始正文内容

如果某篇文章切分后没有可索引内容，系统会清理该文章已有旧向量并跳过写入。

请求体：无（直接 POST）

返回示例：

```json
{
   "synced_count": 5,
   "total_chunks": 23,
   "blog_url": "http://localhost:4321/blog",
   "message": "Blog posts synced successfully"
}
```

#### 8. `POST /ask-blog`

基于已同步的博客内容进行问答。

请求体：

```json
{
   "question": "你的博客里提到了哪些 AI 技术？",
   "top_k": 3
}
```

返回示例：

```json
{
   "question": "你的博客里提到了哪些 AI 技术？",
   "answer": "...",
   "retrieval_method": "vector_search",
   "citations": [
      {
         "source": "Source 1",
         "blog_post_slug": "intro-to-ai",
         "blog_url": "http://localhost:4321/blog/intro-to-ai",
         "chunk_index": 0,
         "distance": 0.35
      }
   ]
}
```

> 前提条件：需要先调用 `/sync-blog` 同步博客内容

## 使用流程

### 文档 RAG 流程

1. 配置 `.env`（至少配置 `DEEPSEEK_API_KEY`）
2. 启动 `uvicorn main:app --reload`
3. 调用 `POST /upload-document` 上传文档
4. 调用 `POST /vector-search-document` 查看召回结果
5. 调用 `POST /ask-document-vector` 做基于文档的问答

### 博客同步和问答流程

1. 配置 `.env`，包括 `DEEPSEEK_API_KEY`、`BLOG_BASE_URL`、`BLOG_URL_PREFIX`
2. 启动 `uvicorn main:app --reload`
3. 调用 `POST /sync-blog` 同步博客内容（首次需要等待一段时间）
4. 调用 `POST /ask-blog` 基于博客内容进行问答

## 目录结构

```text
mini-ai-backend/
├── main.py                    # FastAPI 应用主文件，包含所有接口定义
├── requirements.txt           # Python 依赖
├── README.md                  # 项目说明（此文件）
├── services/                  # 服务模块
│   ├── blog_indexer.py       # 博客索引和同步逻辑
│   ├── blog_load.py          # 加载本地博客文件
│   ├── chunker.py            # 文本切分器
│   └── markdown_cut.py       # Markdown 解析器
├── Content/                   # 项目上下文文档
│   └── AI_Internship_RAG_Project_Context.md
├── chroma_db/                # ChromaDB 向量存储（首次运行自动创建）
└── uploads/                  # 用户上传的文档（首次运行自动创建）
```

## 实现说明

- `main.py` 集成了 FastAPI、DeepSeek API、ChromaDB、文本切分和向量检索
- 支持**两种数据来源**的 RAG：
  - **用户上传文档**：通过 `/upload-document` 上传，存储在 `uploads/` 和 `chroma_db/`
  - **Astro 博客**：通过 `/sync-blog` 同步，自动解析和向量化博客内容
- `services/` 文件夹包含模块化的功能：
  - `blog_indexer.py`：博客同步和索引的核心逻辑
  - `blog_load.py`：从本地或远程加载博客文件
  - `chunker.py`：通用的文本切分逻辑
  - `markdown_cut.py`：Markdown 文件的智能解析（提取标题、段落等结构）
- 所有向量数据都持久化存储在 `chroma_db/`，服务重启不会丢失
- 博客索引会在写入成功后再清理过期旧 chunk，以降低同步失败导致的数据丢失风险

## 小结

这个项目现在是一个"**轻量级 RAG 系统**"，支持：

✅ 基础聊天 (`/chat`)
✅ 文档上传与问答 (`/upload-document`, `/ask-document-vector`)
✅ 向量语义检索 (`/vector-search-document`)
✅ **Astro 博客同步与问答** (`/sync-blog`, `/ask-blog`) ← **新功能**

适合用于：

- RAG 学习项目
- 个人知识库 + 博客的 Q&A 系统

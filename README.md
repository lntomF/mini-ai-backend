# 迷你 AI 后端

现阶段-这是一个基于 FastAPI + DeepSeek + ChromaDB 的小型 AI 后端示例。当前 `main.py` 里实现了 4 个核心能力：基础聊天、文档上传、向量检索、基于检索结果的文档问答。

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
DEEPSEEK_API_KEY=你的key
DEEPSEEK_MODEL=deepseek-v4-flash
```

说明：

- `DEEPSEEK_API_KEY` 必填，否则 `/chat` 和 `/ask-document-vector` 会报错。
- `DEEPSEEK_MODEL` 可选，不配置时默认使用 `deepseek-v4-flash`。

## 启动服务

```bash
uvicorn main:app --reload
```

启动后默认可访问：

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

## 接口速览

### 1. `GET /`

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
      "POST /ask-document-vector"
   ]
}
```

### 2. `GET /health`

返回健康检查信息、已上传文档数量和向量库中的 chunk 数量。

示例响应：

```json
{
   "status": "ok",
   "service": "mini-rag-assistant",
   "document_count": 1,
   "vector_count": 8
}
```

### 3. `POST /chat`

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

### 4. `POST /upload-document`

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

### 5. `POST /vector-search-document`

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

- `doc_id` 为空时，会在所有已上传文档中检索。
- `top_k` 范围是 1 到 10。
- 代码里使用了距离阈值过滤，过远的结果会被丢弃。

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

### 6. `POST /ask-document-vector`

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

## 使用流程

1. 配置 `.env`。
2. 启动 `uvicorn main:app --reload`。
3. 调用 `POST /upload-document` 上传文档。
4. 调用 `POST /vector-search-document` 查看召回结果。
5. 调用 `POST /ask-document-vector` 做基于文档的问答。

## 目录结构

```text
mini-ai-backend/
├── main.py
├── requirements.txt
├── README.md
├── chroma_db/
└── uploads/
```

## 实现说明

- `main.py` 里把 FastAPI、DeepSeek、ChromaDB、文本切分和向量检索都集成在一起了。
- 文档上传后会同时落盘到 `uploads/`，并写入 `chroma_db/`。
- 当前版本没有实现意图分类、工具调用、学习计划、关键词检索等接口，README 里不再保留这些内容。

## 小结

这个项目现在更准确地描述为一个“轻量级文档 RAG 后端”：支持聊天、文档上传、向量检索和基于检索上下文的问答。后续如果你要继续扩展，可以再加关键词检索、多文档管理、流式输出或更完整的 Agent 层。-2026.6.15

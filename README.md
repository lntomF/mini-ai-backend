# 迷你 AI 后端练手项目

这是一个基于 FastAPI + DeepSeek + ChromaDB 的小型 AI 后端示例。项目把聊天、意图分类、文档问答、向量检索、学习计划生成和工具调用都放在一个 `main.py` 里，是一个最小可用的 AI 服务。

## 功能

## 环境要求

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

`DEEPSEEK_MODEL` 不写时，默认就是 `deepseek-v4-flash`。

## 启动服务

```bash
uvicorn main:app --reload
```

启动后可以直接访问：

## 接口速览

### 1. `GET /`

返回服务状态和已开放接口列表。

### 2. `POST /analyze`

请求体：

```json
{
   "text": "Hello world"
}
```

### 3. `POST /chat`

请求体：

```json
{
   "message": "你好，帮我介绍一下这个项目"
}
```

### 4. `POST /classify`

输入一段话，返回类似下面的 JSON：

```json
{
   "intent": "qa",
   "confidence": 0.92,
   "reason": "The user is asking a question."
}
```

### 5. `POST /agent`

支持两种工具：

### 6. `POST /upload-document`

只支持 `.txt` 和 `.md` 文件。上传后会：

### 7. `POST /search-document`

对上传文档做关键词检索。

### 8. `POST /ask-document`

先关键词召回，再交给 DeepSeek 做基于上下文的问答。

### 9. `POST /vector-search-document`

用向量相似度做语义检索。

### 10. `POST /ask-document-vector`

先做向量召回，再让模型基于召回内容回答问题。

### 11. `POST /study-plan`

请求体：

```json
{
   "goal": "学习 FastAPI",
   "days": 7
}
```

### 12. `POST /evaluate-rag`

输入：

```json
{
   "question": "什么是人工智能？",
   "answer": "人工智能是指由计算机系统模拟人类智能的能力。",
   "retrieved_docs": [
      "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
      "人工智能包括机器学习、自然语言处理、计算机视觉等领域。"
   ]
}
```

## 目录结构

```text
mini-ai-backend/
├── main.py
├── requirements.txt
├── README.md
├── chroma_db/
└── uploads/
```

## 说明

## 小结

这个项目是把几个常见 AI 后端能力拼在一起：聊天、分类、工具调用、文档 RAG、向量检索、学习计划生成、RAG 评估。
暂时先学到这几个功能，后续可以继续扩展更多能力，比如多模态、个性化等。-2026.6.15

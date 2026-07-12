from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

HEADING_KEYS = [
    key
    for _, key in HEADERS_TO_SPLIT_ON
]


def split_blog_post(
    post: dict[str, Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 120,
) -> list[dict[str, Any]]:
    """
    将 Markdown 博客按照标题结构和文本长度切分。

    返回的每个 chunk 包含：
    - text：用于 embedding 和检索的增强文本
    - raw_text：原始正文片段
    - heading：标题层级
    - title、slug、tags 等文章元数据
    """

    if not isinstance(post, dict):
        raise TypeError(
            "split_blog_post 需要接收 dict，"
            f"实际收到 {type(post).__name__}"
        )

    content = post.get("content", "")

    if not isinstance(content, str):
        raise TypeError(
            "post['content'] 必须是字符串，"
            f"实际是 {type(content).__name__}"
        )

    content = content.strip()

    if not content:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap 必须大于等于 0，"
            "并且小于 chunk_size"
        )

    title = str(
        post.get("title") or "未命名文章"
    )

    slug = str(
        post.get("slug") or ""
    )

    source_path = str(
        post.get("source_path") or ""
    )

    tags = post.get("tags", [])

    if isinstance(tags, str):
        tags = [tags]
    elif not isinstance(tags, list):
        tags = []

    tags = [
        str(tag)
        for tag in tags
    ]

    # 第一层：按照 Markdown 标题结构切分
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    sections = markdown_splitter.split_text(
        content
    )

    # 第二层：章节太长时进一步切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。", "！", "？",
            "；", "，",
            " ",
            "",
        ],
    )

    docs = text_splitter.split_documents(
        sections
    )

    chunks: list[dict[str, Any]] = []

    for doc in docs:
        raw_text = doc.page_content.strip()

        if not raw_text:
            continue

        heading_parts = [
            str(value).strip()
            for key in HEADING_KEYS
            if (
                value := doc.metadata.get(key)
            )
        ]

        heading = " > ".join(
            heading_parts
        )

        # 把标题与章节加入 embedding 文本，
        # 提升技术名词和章节上下文的召回效果。
        text_parts = [
            f"文章标题：{title}",
        ]

        if heading:
            text_parts.append(
                f"章节：{heading}"
            )

        if tags:
            text_parts.append(
                f"标签：{', '.join(tags)}"
            )

        text_parts.append(
            f"正文：\n{raw_text}"
        )

        retrieval_text = "\n".join(
            text_parts
        )

        chunk_index = len(chunks)

        chunks.append({
            # 用于 embedding 和送给 LLM
            "text": retrieval_text,

            # 便于前端展示原始文章片段
            "raw_text": raw_text,

            "title": title,
            "slug": slug,
            "tags": tags.copy(),
            "heading": heading,
            "chunk_index": chunk_index,
            "source_path": source_path,
        })

    return chunks
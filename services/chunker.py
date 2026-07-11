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
HEADING_KEYS = [key for _, key in HEADERS_TO_SPLIT_ON]


def split_blog_post(post: dict[str, Any]) -> list[dict[str, Any]]:
    """
    将一篇博客文章（parse_markdown_file 的返回结果）按标题层级和字符数
    切分为多个 chunk，供下游做向量化 / 检索使用。

    Args:
        post: 包含 content/title/slug/tags/source_path 等字段的字典，
              通常来自 parse_markdown_file 的返回值。

    Returns:
        chunk 字典列表，每个 chunk 包含切分后的文本及其元信息。
    """
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    sections = markdown_splitter.split_text(post["content"])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    docs = text_splitter.split_documents(sections)

    chunks: list[dict[str, Any]] = []
    for chunk_index, doc in enumerate(docs):
        heading = " > ".join(
            value
            for key in HEADING_KEYS
            if (value := doc.metadata.get(key))
        )
        chunks.append({
            "text": doc.page_content,
            "title": post["title"],
            "slug": post["slug"],
            "tags": list(post["tags"]),  # 拷贝一份，避免多个 chunk 共享同一个 list 引用
            "heading": heading,
            "chunk_index": chunk_index,
            "source_path": post["source_path"],
        })
    return chunks
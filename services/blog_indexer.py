import hashlib
import logging
from pathlib import Path
from typing import Any

from services.blog_load import load_blog_files
from services.markdown_cut import parse_markdown_file
from services.chunker import split_blog_post


logger = logging.getLogger(__name__)


def build_chunk_id(
    slug: str,
    chunk_index: int,
    text: str,
) -> str:
    """
    根据文章 slug、chunk 序号和文本内容生成稳定的 chunk ID。

    同一篇文章内容没有变化时，生成的 ID 不会变化。
    """
    raw = f"{slug}:{chunk_index}:{text}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def normalize_chunks(
    raw_chunks: list[Any],
) -> tuple[list[str], list[str], list[str]]:
    """
    将 chunker 返回的数据统一转换成：

    texts:
        ["chunk 内容 1", "chunk 内容 2"]

    raw_texts:
        ["原始正文 1", "原始正文 2"]

    这里是有意分离的：
    - texts 用于 embedding 和去重标识
    - raw_texts 用于写入 Chroma，保留原始正文内容

    headings:
        ["章节标题 1", "章节标题 2"]

    支持两种 chunker 返回格式：

    1. list[str]

    2. list[dict]
       {
           "text": "...",
           "heading": "..."
       }
    """
    texts: list[str] = []
    raw_texts: list[str] = []
    headings: list[str] = []

    for chunk in raw_chunks:
        if isinstance(chunk, str):
            text = chunk.strip()
            raw_text = text
            heading = ""

        elif isinstance(chunk, dict):
            text = str(
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("page_content")
                or ""
            ).strip()

            raw_text = str(
                chunk.get("raw_text")
                or chunk.get("content")
                or chunk.get("page_content")
                or text
                or ""
            ).strip()

            heading = str(
                chunk.get("heading")
                or chunk.get("section")
                or ""
            ).strip()

        else:
            raise TypeError(
                "split_blog_post 返回了不支持的数据类型："
                f"{type(chunk).__name__}"
            )

        if not text:
            continue

        texts.append(text)
        raw_texts.append(raw_text)
        headings.append(heading)

    return texts, raw_texts, headings


def normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return []

    if isinstance(tags, str):
        candidates = [tags]
    elif isinstance(tags, (list, tuple, set)):
        candidates = tags
    else:
        candidates = [tags]

    normalized = [
        str(tag).strip()
        for tag in candidates
        if tag is not None and str(tag).strip()
    ]

    return list(dict.fromkeys(normalized))


def validate_post(post: Any) -> dict[str, Any]:
    """
    检查并标准化 Markdown 解析结果。
    """
    if not isinstance(post, dict):
        raise TypeError(
            "parse_markdown_file 必须返回 dict，"
            f"实际返回了 {type(post).__name__}"
        )

    required_fields = {
        "slug",
        "title",
        "content",
        "source_path",
    }

    missing_fields = required_fields - post.keys()

    if missing_fields:
        raise ValueError(
            f"文章缺少必要字段：{sorted(missing_fields)}"
        )

    if not isinstance(post["content"], str):
        raise TypeError(
            "post['content'] 必须是字符串，"
            f"实际是 {type(post['content']).__name__}"
        )

    if post["slug"] is None:
        raise ValueError("文章 slug 不能为空")

    if post["title"] is None:
        raise ValueError("文章 title 不能为空")

    if post["source_path"] is None:
        raise ValueError("文章 source_path 不能为空")

    slug = str(post["slug"]).strip().strip("/")
    title = str(post["title"]).strip()
    source_path = str(post["source_path"]).strip()

    if not slug:
        raise ValueError("文章 slug 不能为空")

    if not title:
        raise ValueError("文章 title 不能为空")

    if not source_path:
        raise ValueError("文章 source_path 不能为空")

    post["slug"] = slug
    post["title"] = title
    post["source_path"] = source_path
    post["tags"] = normalize_tags(post.get("tags"))

    return post


def delete_old_blog_chunks(
    vector_collection,
    slug: str,
) -> None:
    """
    删除指定博客文章以前索引的所有 chunk。
    """
    try:
        vector_collection.delete(
            where={
                "$and": [
                    {
                        "source_type": {
                            "$eq": "astro_blog"
                        }
                    },
                    {
                        "slug": {
                            "$eq": slug
                        }
                    },
                ]
            }
        )

    except Exception as exc:
        logger.error(
            "删除文章旧向量失败，slug=%s",
            slug,
        )

        raise RuntimeError(
            f"删除文章旧向量失败，slug={slug}: {exc}"
        ) from exc


def index_blog_post(
    post: dict[str, Any],
    vector_collection,
    embedding_model,
    blog_base_url: str,
    blog_url_prefix: str,
) -> int:
    """
    将一篇 Astro Markdown 博客文章写入 Chroma。

    流程：

    1. 校验文章数据
    2. 根据 Markdown 内容切分 chunk
    3. 生成 embedding
    4. 写入 Chroma
    5. 清理过期的旧 chunk
    """
    post = validate_post(post)

    # 注意：这里传入完整 post 字典
    # 不要写成 split_blog_post(post["content"])
    raw_chunks = split_blog_post(post)

    if not isinstance(raw_chunks, list):
        raise TypeError(
            "split_blog_post 必须返回 list，"
            f"实际返回了 {type(raw_chunks).__name__}"
        )

    chunks, raw_texts, chunk_headings = normalize_chunks(
        raw_chunks
    )

    # chunks 是 embedding/去重的稳定输入，raw_texts 是展示用的原始正文。
    # 两者刻意分离，避免检索文本和落库文本混用。

    if not chunks:
        delete_old_blog_chunks(
            vector_collection=vector_collection,
            slug=post["slug"],
        )

        logger.info(
            "文章无可索引内容，已清理旧数据：slug=%s",
            post["slug"],
        )
        return 0

    url = (
        f"{blog_base_url.rstrip('/')}/"
        f"{blog_url_prefix.strip('/')}/"
        f"{post['slug']}"
    )

    chunk_ids = [
        build_chunk_id(
            slug=post["slug"],
            chunk_index=index,
            text=chunk,
        )
        for index, chunk in enumerate(chunks)
    ]

    metadatas = [
        {
            "source_type": "astro_blog",
            "slug": post["slug"],
            "title": post["title"],
            "url": url,
            "filename": Path(
                post["source_path"]
            ).name,
            "source_path": post["source_path"],
            "chunk_index": index,
            "heading": chunk_headings[index],
            "tags": ",".join(post["tags"]),
        }
        for index in range(len(chunks))
    ]

    try:
        embeddings = embedding_model.encode(
            chunks
        ).tolist()

    except Exception as exc:
        logger.error(
            "生成 embedding 失败：slug=%s",
            post["slug"],
        )

        raise RuntimeError(
            f"生成 embedding 失败，slug={post['slug']}: {exc}"
        ) from exc

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding 数量与 chunk 数量不一致："
            f"chunks={len(chunks)}, "
            f"embeddings={len(embeddings)}"
        )

    existing_ids: list[str] = []

    try:
        existing_chunks = vector_collection.get(
            where={
                "$and": [
                    {
                        "source_type": {
                            "$eq": "astro_blog"
                        }
                    },
                    {
                        "slug": {
                            "$eq": post["slug"]
                        }
                    },
                ]
            }
        )

        existing_ids = [
            str(chunk_id)
            for chunk_id in existing_chunks.get("ids") or []
        ]

    except Exception as exc:
        logger.warning(
            "读取旧向量失败，跳过旧 chunk 清理，slug=%s, error=%s",
            post["slug"],
            exc,
        )

    try:
        vector_collection.upsert(
            ids=chunk_ids,
            documents=raw_texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    except Exception as exc:
        logger.exception(
            "写入 Chroma 失败：slug=%s",
            post["slug"],
        )

        raise RuntimeError(
            f"写入 Chroma 失败，slug={post['slug']}: {exc}"
        ) from exc

    stale_ids = [
        chunk_id
        for chunk_id in existing_ids
        if chunk_id not in chunk_ids
    ]

    if stale_ids:
        try:
            vector_collection.delete(
                ids=stale_ids,
            )

        except Exception as exc:
            logger.warning(
                "清理旧 chunk 失败，slug=%s, stale_count=%d, error=%s",
                post["slug"],
                len(stale_ids),
                exc,
            )

    logger.info(
        "文章索引完成：slug=%s，chunk_count=%d",
        post["slug"],
        len(chunks),
    )

    return len(chunks)


def sync_all_blog_posts(
    vector_collection,
    embedding_model,
    blog_base_url: str,
    blog_url_prefix: str,
) -> dict[str, Any]:
    """
    扫描并同步全部 Astro 博客文章。

    单篇博客失败时不会中断其他文章，
    错误信息会被放进 failed_posts。
    """
    files = load_blog_files()

    synced_posts: list[dict[str, Any]] = []
    failed_posts: list[dict[str, str]] = []

    logger.info(
        "开始同步博客，发现文件数量：%d",
        len(files),
    )

    for file_path in files:
        try:
            post = parse_markdown_file(
                file_path
            )

            logger.info(
                "开始处理博客：%s",
                file_path,
            )

            chunk_count = index_blog_post(
                post=post,
                vector_collection=vector_collection,
                embedding_model=embedding_model,
                blog_base_url=blog_base_url,
                blog_url_prefix=blog_url_prefix,
            )

            synced_posts.append({
                "slug": post["slug"],
                "title": post["title"],
                "file": str(file_path),
                "chunk_count": chunk_count,
            })

        except Exception as exc:
            logger.exception(
                "处理博客文件失败：%s",
                file_path,
            )

            failed_posts.append({
                "file": str(file_path),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            })

    result = {
        "file_count": len(files),
        "synced_count": len(synced_posts),
        "failed_count": len(failed_posts),
        "synced_posts": synced_posts,
        "failed_posts": failed_posts,
    }

    logger.info(
        "博客同步结束：总数=%d，成功=%d，失败=%d",
        result["file_count"],
        result["synced_count"],
        result["failed_count"],
    )

    return result
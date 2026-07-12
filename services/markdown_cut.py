from pathlib import Path
from typing import Any, Optional
import logging

import frontmatter


logger = logging.getLogger(__name__)


def parse_markdown_file(
    path: Path,
    required_fields: Optional[list[str]] = None,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if not path.is_file():
        raise ValueError(f"路径不是文件: {path}")

    try:
        post = frontmatter.load(
            path,
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(
            "解析 Markdown 文件失败: %s",
            path,
            exc_info=True,
        )
        raise ValueError(
            f"解析 Markdown 文件失败 {path}: {exc}"
        ) from exc

    metadata = post.metadata or {}

    if not isinstance(metadata, dict):
        raise ValueError(
            f"文件 {path} 的 frontmatter 格式不正确（应为字典）"
        )

    slug = str(
        metadata.get("slug") or path.stem
    ).strip("/")

    title = str(
        metadata.get("title")
        or slug.replace("-", " ").title()
    )

    if required_fields:
        missing_fields = [
            field
            for field in required_fields
            if field not in metadata
            or metadata.get(field) in (None, "")
        ]

        if missing_fields:
            raise ValueError(
                f"文件 {path} 缺少必需字段: "
                f"{', '.join(missing_fields)}"
            )

    tags = metadata.get("tags", [])

    if isinstance(tags, str):
        tags = [
            tag.strip()
            for tag in tags.split(",")
            if tag.strip()
        ]
    elif isinstance(tags, list):
        tags = [
            str(tag)
            for tag in tags
        ]
    else:
        tags = []

    date = (
        metadata.get("date")
        or metadata.get("pubDate")
        or metadata.get("publishedAt")
    )

    if date and hasattr(date, "isoformat"):
        date = date.isoformat()
    elif date is not None:
        date = str(date)

    description = str(
        metadata.get("description") or ""
    )

    draft = bool(
        metadata.get("draft") or False
    )

    return {
        "slug": slug,
        "title": title,
        "description": description,
        "tags": tags,
        "date": date,
        "draft": draft,
        "content": (
            post.content.strip()
            if post.content
            else ""
        ),
        "source_path": str(path),
        "file_name": path.name,
    }


def validate_markdown_file(
    path: Path,
) -> tuple[bool, Optional[str]]:
    try:
        parse_markdown_file(path)
        return True, None

    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)
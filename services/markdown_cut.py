from pathlib import Path
from typing import Any, Optional
import logging
import frontmatter

logger = logging.getLogger(__name__)


def parse_markdown_file(
    path: Path,
    required_fields: Optional[list[str]] = None
) -> dict[str, Any]:
    """
    解析 Markdown 文件，提取 frontmatter 和内容。

    Args:
        path: Markdown 文件路径
        required_fields: 必需的 frontmatter 字段列表，如果缺少则抛出异常

    Returns:
        包含解析结果的字典

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 缺少必需字段或解析失败
    """
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if not path.is_file():
        raise ValueError(f"路径不是文件: {path}")

    try:
        post = frontmatter.load(path, encoding="utf-8")
    except Exception as e:
        logger.warning("解析 Markdown 文件失败: %s", path, exc_info=True)
        raise ValueError(f"解析 Markdown 文件失败 {path}: {e}") from e

    # 获取 metadata，确保不为 None，且必须是字典
    metadata = post.metadata or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"文件 {path} 的 frontmatter 格式不正确（应为字典）")

    # 使用更明确的变量名
    slug = metadata.get("slug") or path.stem
    title = metadata.get("title") or slug.replace("-", " ").title()

    # 验证必需字段
    if required_fields:
        missing_fields = [field for field in required_fields if field not in metadata]
        if missing_fields:
            raise ValueError(
                f"文件 {path} 缺少必需字段: {', '.join(missing_fields)}"
            )

    # 处理 tags，确保是列表
    tags = metadata.get("tags", [])
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    elif not isinstance(tags, list):
        tags = []

    # 获取日期（如果存在）
    date = metadata.get("date")
    if date and hasattr(date, "isoformat"):
        date = date.isoformat()

    # description / draft：显式处理值为 None 的情况（.get 的 default 只在 key 不存在时生效）
    description = metadata.get("description") or ""
    draft = bool(metadata.get("draft") or False)

    return {
        "slug": slug,
        "title": title,
        "description": description,
        "tags": tags,
        "date": date,
        "draft": draft,
        "content": post.content.strip() if post.content else "",
        "source_path": str(path),
        "file_name": path.name,
    }


def validate_markdown_file(path: Path) -> tuple[bool, Optional[str]]:
    """
    验证 Markdown 文件是否有效（不抛出异常）。

    Args:
        path: Markdown 文件路径

    Returns:
        (是否有效, 错误信息)
    """
    try:
        parse_markdown_file(path)
        return True, None
    except (FileNotFoundError, ValueError) as e:
        return False, str(e)
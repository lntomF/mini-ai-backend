from pathlib import Path

# 相对于当前文件所在目录
# 当前文件: G:/FOR STUDY/AI Agent/mini-ai-backend/services/xxx.py
# 目标目录: G:/FOR STUDY/AI Agent/fumbling-field/src/content/blog

BLOG_DIRECTORY = Path(__file__).resolve().parents[2] / "fumbling-field" / "src" / "content" / "blog"

def load_blog_files() -> list[Path]:
    if not BLOG_DIRECTORY.exists():
        raise FileNotFoundError(
            f"博客目录不存在: {BLOG_DIRECTORY}"
        )

    files = [
        *BLOG_DIRECTORY.rglob("*.md"),
        *BLOG_DIRECTORY.rglob("*.mdx"),
    ]

    return sorted(files)
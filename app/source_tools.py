from __future__ import annotations

import re
from html import unescape
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.tools import tool
from openai import OpenAI

from .parsers import VIDEO_EXTENSIONS, parse_docx, parse_pdf, parse_pptx, parse_video


SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".docx", ".txt", ".rtf", *VIDEO_EXTENSIONS}


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[\\s\\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\\s\\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return " ".join(text.split())


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _download_url_text(url: str, timeout_s: int = 20, max_chars: int = 10000) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
        content_type = (response.headers.get("content-type") or "").lower()

    text = raw.decode("utf-8", errors="ignore")
    if "text/html" in content_type:
        text = _strip_html(text)
    else:
        text = " ".join(text.split())
    return text[:max_chars]


def _extract_local_source(path: Path) -> tuple[bool, str]:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_SOURCE_EXTENSIONS:
        raise ValueError(f"Unsupported source file type: {ext}")

    if ext == ".docx":
        text, _ = parse_docx(path)
    elif ext == ".pdf":
        text, _ = parse_pdf(path)
    elif ext == ".pptx":
        text, _ = parse_pptx(path)
    elif ext in VIDEO_EXTENSIONS:
        text, _ = parse_video(path, OpenAI())
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    return True, text[:10000]


def _extract_source_item(item: str) -> dict[str, Any]:
    candidate = item.strip()
    if not candidate:
        return {"source": item, "ok": False, "error": "Empty source entry", "content": ""}

    try:
        if _is_http_url(candidate):
            return {
                "source": candidate,
                "type": "url",
                "ok": True,
                "content": _download_url_text(candidate),
            }

        path = Path(candidate)
        if path.exists() and path.is_file():
            ok, content = _extract_local_source(path)
            return {
                "source": candidate,
                "type": "file",
                "ok": ok,
                "content": content,
            }

        return {
            "source": candidate,
            "type": "plain_text",
            "ok": True,
            "content": candidate[:10000],
        }
    except (ValueError, URLError, TimeoutError, OSError) as exc:
        return {"source": candidate, "ok": False, "error": str(exc), "content": ""}


@tool("fetch_sources_context")
def fetch_sources_context(items: list[str]) -> list[dict[str, Any]]:
    """Fetch source context from URLs, local files (.pdf/.docx/.txt/.rtf/video), or raw text entries."""
    return [_extract_source_item(item) for item in items]


def build_sources_context(items: list[str]) -> str:
    fetched = fetch_sources_context.invoke({"items": items})
    blocks: list[str] = []
    for item in fetched:
        source_type = item.get("type", "unknown")
        if item["ok"]:
            blocks.append(
                f"Source ({source_type}): {item['source']}\nExtracted Content:\n{item['content']}"
            )
        else:
            blocks.append(f"Source ({source_type}): {item['source']}\nError: {item['error']}")
    return "\n\n".join(blocks)

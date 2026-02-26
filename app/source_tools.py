from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from langchain_core.tools import tool


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[\\s\\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\\s\\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return " ".join(text.split())


def _download_url_text(url: str, timeout_s: int = 15, max_chars: int = 8000) -> str:
    request = Request(url, headers={"User-Agent": "GraderBot/1.0"})
    with urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
        content_type = (response.headers.get("content-type") or "").lower()

    text = raw.decode("utf-8", errors="ignore")
    if "text/html" in content_type:
        text = _strip_html(text)
    else:
        text = " ".join(text.split())
    return text[:max_chars]


@tool("fetch_sources_context")
def fetch_sources_context(urls: list[str]) -> list[dict[str, Any]]:
    """Fetch URL sources and return extracted textual context per source."""
    results: list[dict[str, Any]] = []
    for url in urls:
        try:
            text = _download_url_text(url)
            results.append({"url": url, "ok": True, "content": text})
        except (ValueError, URLError, TimeoutError, OSError) as exc:
            results.append({"url": url, "ok": False, "error": str(exc), "content": ""})
    return results


def build_sources_context(urls: list[str]) -> str:
    fetched = fetch_sources_context.invoke(urls)
    blocks: list[str] = []
    for item in fetched:
        if item["ok"]:
            blocks.append(f"Source URL: {item['url']}\nExtracted Content:\n{item['content']}")
        else:
            blocks.append(f"Source URL: {item['url']}\nError: {item['error']}")
    return "\n\n".join(blocks)

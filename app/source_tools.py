from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests

from langchain_core.tools import tool
from openai import OpenAI

from parsers import VIDEO_EXTENSIONS, parse_docx, parse_pdf, parse_pptx, parse_video


SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".docx", ".txt", ".rtf", *VIDEO_EXTENSIONS}

BLOCKED_PAGE_MARKERS = (
    "access denied",
    "you don't have permission to access"
)

MAX_CHARS = 2000000

def _strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(strip=True, separator=" ")
    #text = re.sub(r"<script[\\s\\S]*?</script>", " ", html, flags=re.IGNORECASE)
    #text = re.sub(r"<style[\\s\\S]*?</style>", " ", text, flags=re.IGNORECASE)
    #text = re.sub(r"<[^>]+>", " ", text)
    #text = unescape(text)
    #return " ".join(text.split())


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

def _looks_like_block_page(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return all(marker in normalized for marker in BLOCKED_PAGE_MARKERS)

def _get_playwright_storage_state_path() -> str | None:
    candidate = os.getenv("PLAYWRIGHT_STORAGE_STATE_PATH", "").strip()
    return candidate or None

def _playwright_headless() -> bool:
    configured = os.getenv("PLAYWRIGHT_HEADLESS", "true").strip().lower()
    return configured not in {"0", "false", "no"}


def _download_url_text(url: str, timeout_s: int = 20, max_chars: int = MAX_CHARS) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Origin": f"{urlparse(url).scheme}://{urlparse(url).netloc}",
        "Connection": "keep-alive",
    }

    def _normalize_text(raw_text: str, content_type: str) -> str:
        if "text/html" in content_type:
            text = _strip_html(raw_text)
        else:
            text = " ".join(raw_text.split())
        return text[:max_chars]

    # First attempt: keep a lightweight HTTP request path for simple sources.
    try:
        response = requests.get(url, headers=headers, timeout=timeout_s, allow_redirects=True)
        response.raise_for_status()
        text = _normalize_text(response.text, (response.headers.get("content-type") or "").lower())

        if not _looks_like_block_page(text):
            return text
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 403:
            raise URLError(str(exc)) from exc
    except requests.RequestException as exc:
        raise URLError(str(exc)) from exc

    # Fallback for anti-bot protected pages that need full browser execution.
    return _download_url_text_with_playwright(
        url=url,
        timeout_s=timeout_s,
        max_chars=max_chars,
        storage_state_path=_get_playwright_storage_state_path(),
    )

def _download_url_text_with_playwright(
    url: str,
    timeout_s: int = 20,
    max_chars: int = MAX_CHARS,
    storage_state_path: str | None = None,
    headless: bool | None = None,
) -> str:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - dependency should be installed via pyproject.
        raise URLError("Playwright is not installed; cannot process anti-bot protected URL") from exc

    browser_timeout_ms = timeout_s * 1000
    if headless is None:
        headless = _playwright_headless()

    context_kwargs: dict[str, Any] = {
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "locale": "en-US",
    }
    if storage_state_path:
        context_kwargs["storage_state"] = storage_state_path

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(**context_kwargs)
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=browser_timeout_ms)
            for selector in [
                "#onetrust-accept-btn-handler",
                "button:has-text('Accept')",
                "button:has-text('I Agree')",
            ]:
                try:
                    page.locator(selector).first.click(timeout=1500)
                    break
                except PlaywrightTimeoutError:
                    continue

            page.wait_for_timeout(2000)
            html = page.content()
        except PlaywrightTimeoutError as exc:
            raise URLError(f"Timed out while rendering URL in browser: {url}") from exc
        finally:
            context.close()
            browser.close()

    text = _strip_html(html)[:max_chars]
    if _looks_like_block_page(text):
        raise URLError(
            "Remote site returned an anti-bot/access-denied page; unable to extract article text. "
            "Try running from a browser-authenticated network or provide a local file copy."
        )
    return text

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
        text = _strip_html(text)

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
            "content": candidate[:MAX_CHARS],
        }
    except (ValueError, URLError, TimeoutError, OSError) as exc:
        return {"source": candidate, "ok": False, "error": str(exc), "content": ""}


@tool("fetch_sources_context")
def fetch_sources_context(items: list[str]) -> list[dict[str, Any]]:
    """Fetch source context from URLs, local files (.pdf/.docx/.txt/.rtf/video), or raw text entries."""
    return [_extract_source_item(item) for item in items]


def build_sources_context(items: list[str]) -> tuple[str, list[bool]]:
    fetched = fetch_sources_context.invoke({"items": items})
    blocks: list[str] = []
    status: list[bool] = []
    for item in fetched:
        source_type = item.get("type", "unknown")
        if item["ok"]:
            blocks.append(
                f"Source ({source_type}): {item['source']}\nExtracted Content:\n{item['content']}"
            )
            status.append(True)
        else:
            blocks.append(f"Source ({source_type}): {item['source']}\nError: {item['error']}")
            status.append(False)
    return "\n\n".join(blocks), status

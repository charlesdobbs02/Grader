from __future__ import annotations

import requests


def test_build_sources_context_invokes_tool_with_named_argument(monkeypatch) -> None:
    from app import source_tools

    called = {}

    def fake_invoke(payload):
        called["payload"] = payload
        return [
            {"source": "https://example.com/a", "type": "url", "ok": True, "content": "Alpha"},
            {"source": "notes text", "type": "plain_text", "ok": False, "error": "boom", "content": ""},
        ]

    class FakeTool:
        @staticmethod
        def invoke(payload):
            return fake_invoke(payload)

    monkeypatch.setattr(source_tools, "fetch_sources_context", FakeTool())

    rendered, status = source_tools.build_sources_context([
        "https://example.com/a",
        "notes text",
    ])

    assert called["payload"] == {
        "items": ["https://example.com/a", "notes text"]
    }
    assert "Source (url): https://example.com/a" in rendered
    assert "Extracted Content:\nAlpha" in rendered
    assert "Source (plain_text): notes text" in rendered
    assert "Error: boom" in rendered
    assert status == [True, False]


def test_download_url_text_uses_playwright_fallback_on_403(monkeypatch) -> None:
    from app import source_tools

    class FakeResponse:
        status_code = 403

    def fake_get(*args, **kwargs):
        raise requests.HTTPError("forbidden", response=FakeResponse())

    monkeypatch.setattr(source_tools.requests, "get", fake_get)
    monkeypatch.setattr(
        source_tools,
        "_download_url_text_with_playwright",
        lambda url, timeout_s, max_chars, **kwargs: "rendered content",
    )

    text = source_tools._download_url_text("https://example.com/blocked")

    assert text == "rendered content"


def test_download_url_text_uses_playwright_fallback_on_block_page(monkeypatch) -> None:
    from app import source_tools

    class FakeResponse:
        headers = {"content-type": "text/html"}
        text = (
            "Access Denied You don't have permission to access this server. "
            "Reference #18.69cfdb17.1773613410.7d39c944 https://errors.edgesuite.net/"
        )

        @staticmethod
        def raise_for_status() -> None:
            return None

    monkeypatch.setattr(source_tools.requests, "get", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(
        source_tools,
        "_download_url_text_with_playwright",
        lambda url, timeout_s, max_chars, **kwargs: "rendered content",
    )

    text = source_tools._download_url_text("https://example.com/blocked")

    assert text == "rendered content"


def test_extract_source_item_returns_error_for_playwright_block_page(monkeypatch) -> None:
    from app import source_tools

    monkeypatch.setattr(
        source_tools,
        "_download_url_text_with_playwright",
        lambda url, timeout_s, max_chars, **kwargs: (_ for _ in ()).throw(
            source_tools.URLError("Remote site returned an anti-bot/access-denied page")
        ),
    )

    class FakeResponse:
        headers = {"content-type": "text/html"}
        text = (
            "Access Denied You don't have permission to access this server. "
            "Reference #18.69cfdb17.1773613410.7d39c944 https://errors.edgesuite.net/"
        )

        @staticmethod
        def raise_for_status() -> None:
            return None

    monkeypatch.setattr(source_tools.requests, "get", lambda *args, **kwargs: FakeResponse())

    result = source_tools._extract_source_item("https://example.com/blocked")

    assert result["ok"] is False
    assert "anti-bot/access-denied" in result["error"]


def test_playwright_storage_state_path_from_env(monkeypatch) -> None:
    from app import source_tools

    monkeypatch.setenv("PLAYWRIGHT_STORAGE_STATE_PATH", " /tmp/state.json ")

    assert source_tools._get_playwright_storage_state_path() == "/tmp/state.json"


def test_download_url_text_passes_storage_state_to_playwright(monkeypatch) -> None:
    from app import source_tools

    class FakeResponse:
        status_code = 403

    captured = {}

    def fake_get(*args, **kwargs):
        raise requests.HTTPError("forbidden", response=FakeResponse())

    def fake_fallback(url, timeout_s, max_chars, storage_state_path=None, headless=None):
        captured["storage_state_path"] = storage_state_path
        return "rendered content"

    monkeypatch.setenv("PLAYWRIGHT_STORAGE_STATE_PATH", "/tmp/state.json")
    monkeypatch.setattr(source_tools.requests, "get", fake_get)
    monkeypatch.setattr(source_tools, "_download_url_text_with_playwright", fake_fallback)

    text = source_tools._download_url_text("https://example.com/blocked")

    assert text == "rendered content"
    assert captured["storage_state_path"] == "/tmp/state.json"
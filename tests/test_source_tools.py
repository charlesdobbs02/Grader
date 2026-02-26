from __future__ import annotations


def test_build_sources_context_invokes_tool_with_named_argument(monkeypatch) -> None:
    from app import source_tools

    called = {}

    def fake_invoke(payload):
        called["payload"] = payload
        return [
            {"source": "https://example.com/a", "type": "url", "ok": True, "content": "Alpha"},
            {"source": "notes text", "type": "plain_text", "ok": False, "error": "boom", "content": ""},
        ]

    monkeypatch.setattr(source_tools.fetch_sources_context, "invoke", fake_invoke)

    rendered = source_tools.build_sources_context([
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

from __future__ import annotations

from pathlib import Path


def test_parse_docx_appends_embedded_image_descriptions(monkeypatch, tmp_path: Path) -> None:
    from app import parsers

    fake_path = tmp_path / "submission.docx"
    fake_path.write_bytes(b"fake")

    class _Paragraph:
        def __init__(self, text: str) -> None:
            self.text = text

    class _TargetPart:
        blob = b"image-bytes"

    class _Rel:
        reltype = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        target_part = _TargetPart()

    class _Part:
        rels = {"r1": _Rel()}

    class FakeDocument:
        def __init__(self, _path: Path) -> None:
            self.paragraphs = [_Paragraph("Paragraph text")]
            self.part = _Part()

    monkeypatch.setattr(parsers, "Document", FakeDocument)
    monkeypatch.setattr(
        parsers,
        "_describe_images_with_openai",
        lambda images, client: ["[Image 1] A chart showing growth over time."],
    )

    text, metadata = parsers.parse_docx(fake_path)

    assert "Paragraph text" in text
    assert "Embedded Images:" in text
    assert "A chart showing growth over time" in text
    assert metadata["image_count"] == 1


def test_parse_pdf_appends_embedded_image_descriptions(monkeypatch, tmp_path: Path) -> None:
    from app import parsers

    fake_path = tmp_path / "submission.pdf"
    fake_path.write_bytes(b"fake")

    class _Image:
        data = b"image-bytes"

    class _Page:
        images = [_Image()]

        @staticmethod
        def extract_text() -> str:
            return "Page text"

    class FakePdfReader:
        def __init__(self, _path: str) -> None:
            self.pages = [_Page()]

    monkeypatch.setattr(parsers, "PdfReader", FakePdfReader)
    monkeypatch.setattr(
        parsers,
        "_describe_images_with_openai",
        lambda images, client: ["[Image 1] Screenshot of a rubric table."],
    )

    text, metadata = parsers.parse_pdf(fake_path)

    assert "Page text" in text
    assert "Embedded Images:" in text
    assert "rubric table" in text
    assert metadata["image_count"] == 1
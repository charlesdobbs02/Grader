from __future__ import annotations

from pathlib import Path


TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".py", ".html"}


def parse_submission(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix in TEXT_EXTENSIONS:
        return file_path.read_text(errors="ignore")

    if suffix in {".pdf", ".doc", ".docx", ".ppt", ".pptx"}:
        return (
            f"Binary office document received ({file_path.name}). "
            "Integrate a dedicated parser (PyMuPDF, python-docx, python-pptx) in production."
        )

    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return (
            f"Video file received ({file_path.name}). "
            "Integrate speech-to-text/transcript extraction in production."
        )

    return f"Unsupported file type {suffix or '[none]'} for deep parsing."

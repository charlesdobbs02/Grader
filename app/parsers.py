from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from docx import Document
from moviepy import VideoFileClip
from openai import OpenAI
from pypdf import PdfReader
from pptx import Presentation

from models import ParseResult


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def parse_docx(path: Path) -> tuple[str, dict[str, Any]]:
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return text, {"paragraph_count": len(doc.paragraphs)}


def parse_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip(), {"page_count": len(reader.pages)}


def parse_pptx(path: Path) -> tuple[str, dict[str, Any]]:
    presentation = Presentation(str(path))
    lines: list[str] = []
    for i, slide in enumerate(presentation.slides, start=1):
        lines.append(f"[Slide {i}]")
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                lines.append(shape.text.strip())
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                lines.append("[Speaker Notes]")
                lines.append(notes)
    return "\n".join(lines), {"slide_count": len(presentation.slides)}


def parse_video(path: Path, client: OpenAI) -> tuple[str, dict[str, Any]]:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
        temp_audio_path = Path(temp_audio.name)

    try:
        clip = VideoFileClip(str(path))
        duration = clip.duration
        clip.audio.write_audiofile(str(temp_audio_path), logger=None)
        clip.close()

        with temp_audio_path.open("rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file,
            )

        return transcript.text.strip(), {"duration_seconds": duration}
    finally:
        if temp_audio_path.exists():
            os.remove(temp_audio_path)


def detect_assignment_type(file_extension: str) -> str:
    ext = file_extension.lower()
    if ext in {".pdf", ".doc", ".docx", ".txt", ".rtf"}:
        return "Written Assignment"
    if ext in {".ppt", ".pptx"}:
        return "PowerPoint"
    if ext in VIDEO_EXTENSIONS:
        return "Video"
    raise ValueError(f"Unsupported submission type: {ext}")


def parse_submission_file(path: Path, openai_client: OpenAI) -> ParseResult:
    ext = path.suffix.lower()
    assignment_type = detect_assignment_type(ext)

    if ext in {".docx"}:
        text, metadata = parse_docx(path)
    elif ext in {".pdf"}:
        text, metadata = parse_pdf(path)
    elif ext in {".pptx"}:
        text, metadata = parse_pptx(path)
    elif ext in VIDEO_EXTENSIONS:
        text, metadata = parse_video(path, openai_client)
    elif ext in {".txt", ".rtf", ".doc", ".ppt"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        metadata = {"fallback_parser": True}
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if not text.strip():
        raise ValueError("No extractable content found in submission")

    return ParseResult(
        assignment_type=assignment_type,
        extracted_text=text,
        parser_metadata=metadata,
    )

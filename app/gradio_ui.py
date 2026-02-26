from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
from openai import OpenAI

from .agents import GradingOrchestrator
from .parsers import parse_submission_file

RUBRIC_PATH = Path("data/rubrics.json")


def _format_result(result: Any) -> str:
    if hasattr(result, "model_dump"):
        payload = result.model_dump()
    else:
        payload = result
    return json.dumps(payload, indent=2)


def grade_submission_ui(
    submission_file: str,
    assignment_instructions: str,
    provided_sources: str,
) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not configured."

    if not submission_file:
        return "Please upload a submission file."

    if not assignment_instructions.strip():
        return "Please provide assignment instructions."

    client = OpenAI()
    temp_path: Path | None = None

    try:
        original_path = Path(submission_file)
        suffix = original_path.suffix

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(original_path.read_bytes())
            temp_path = Path(temp.name)

        parsed = parse_submission_file(temp_path, openai_client=client)
        orchestrator = GradingOrchestrator(rubric_path=RUBRIC_PATH)
        sources = [line.strip() for line in provided_sources.splitlines() if line.strip()]

        result = orchestrator.grade_submission(
            assignment_type=parsed.assignment_type,
            assignment_text=parsed.extracted_text,
            assignment_instructions=assignment_instructions,
            provided_sources=sources,
        )
        return _format_result(result)
    except Exception as exc:  # surface human-readable errors to the UI
        return f"Error while grading submission: {exc}"
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Agentic Assignment Grader") as demo:
        gr.Markdown("# Agentic Assignment Grader")
        gr.Markdown(
            "Upload a student submission, provide assignment instructions, and list allowed sources."
        )

        with gr.Row():
            with gr.Column(scale=1):
                submission_file = gr.File(
                    label="Student Submission",
                    file_types=[
                        ".docx",
                        ".pdf",
                        ".pptx",
                        ".txt",
                        ".rtf",
                        ".doc",
                        ".ppt",
                        ".mp4",
                        ".mov",
                        ".avi",
                        ".mkv",
                        ".webm",
                        ".m4v",
                    ],
                    type="filepath",
                )
                assignment_instructions = gr.Textbox(
                    label="Assignment Instructions",
                    lines=10,
                    placeholder="Paste assignment instructions here...",
                )
                provided_sources = gr.Textbox(
                    label="Provided Sources (one per line)",
                    lines=8,
                    placeholder="Source 1\nSource 2\nSource 3",
                )
                grade_button = gr.Button("Grade Submission", variant="primary")
            with gr.Column(scale=1):
                output = gr.Code(
                    label="Grading Result (JSON)",
                    language="json",
                    interactive=False,
                )

        grade_button.click(
            fn=grade_submission_ui,
            inputs=[submission_file, assignment_instructions, provided_sources],
            outputs=output,
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()

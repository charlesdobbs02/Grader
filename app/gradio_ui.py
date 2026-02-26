from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
from openai import OpenAI

from agents import GradingOrchestrator
from parsers import parse_submission_file
from workflow_runner import build_markdown_report

RUBRIC_PATH = Path("../data/rubrics.json")


def _normalize_url_sources(provided_sources: str) -> list[str]:
    return [line.strip() for line in provided_sources.splitlines() if line.strip()]


def _source_upload_to_entry(source_file: str, client: OpenAI) -> str:
    source_path = Path(source_file)
    parsed_source = parse_submission_file(source_path, openai_client=client)
    source_name = source_path.name
    return f"Uploaded Source ({source_name}):\n{parsed_source.extracted_text}"


def _build_sources(url_text: str, source_files: list[str] | None, client: OpenAI) -> list[str]:
    sources = _normalize_url_sources(url_text)
    for source_file in source_files or []:
        sources.append(_source_upload_to_entry(source_file, client))
    return sources


def grade_submission_ui(
    submission_files: list[str] | None,
    assignment_instructions: str,
    source_urls: str,
    source_files: list[str] | None,
) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not configured."

    if not submission_files:
        return "Please upload at least one submission file."

    if not assignment_instructions.strip():
        return "Please provide assignment instructions."

    client = OpenAI()
    temp_paths: list[Path] = []

    try:
        orchestrator = GradingOrchestrator(rubric_path=RUBRIC_PATH)
        sources = _build_sources(source_urls, source_files, client)
        graded_results: list[dict[str, Any]] = []

        for submission_file in submission_files:
            original_path = Path(submission_file)
            suffix = original_path.suffix

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(original_path.read_bytes())
                temp_path = Path(temp.name)
                temp_paths.append(temp_path)

            parsed = parse_submission_file(temp_path, openai_client=client)
            result = orchestrator.grade_submission(
                assignment_type=parsed.assignment_type,
                assignment_text=parsed.extracted_text,
                assignment_instructions=assignment_instructions,
                provided_sources=sources,
            )
            graded_results.append(
                {
                    "student_name": original_path.stem,
                    "submission_file": original_path.name,
                    "result": result.model_dump(),
                }
            )

        report_data = {
            "assignment_instructions": assignment_instructions,
            "sources": sources,
            "source_proofs": [],
            "results": graded_results,
        }
        return build_markdown_report(report_data)
    except Exception as exc:  # surface human-readable errors to the UI
        return f"Error while grading submission: {exc}"
    finally:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Agentic Assignment Grader") as demo:
        gr.Markdown("# Agentic Assignment Grader")
        gr.Markdown(
            "Upload one or more student submissions, provide assignment instructions, and supply allowed sources via URLs or source file uploads."
        )

        with gr.Row():
            with gr.Column(scale=1):
                submission_files = gr.File(
                    label="Student Submissions",
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
                    file_count="multiple",
                    type="filepath",
                )
                assignment_instructions = gr.Textbox(
                    label="Assignment Instructions",
                    lines=10,
                    placeholder="Paste assignment instructions here...",
                )
                source_urls = gr.Textbox(
                    label="Provided Source URLs (one URL per line)",
                    lines=8,
                    placeholder="https://example.com/source-1\nhttps://example.com/source-2",
                )
                source_files = gr.File(
                    label="Provided Source Files",
                    file_count="multiple",
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
                grade_button = gr.Button("Grade Submission", variant="primary")
            with gr.Column(scale=1):
                output = gr.Markdown(label="Last Request Markdown Report")

        grade_button.click(
            fn=grade_submission_ui,
            inputs=[submission_files, assignment_instructions, source_urls, source_files],
            outputs=output,
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()

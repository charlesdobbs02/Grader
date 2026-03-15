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
from workflow_runner import build_student_result_lines
from source_tools import build_sources_context

if __debug__:
    RUBRIC_PATH = Path("data/rubrics.json")
else:
    RUBRIC_PATH = Path("../data/rubrics.json")


def _normalize_url_sources(provided_sources: str) -> list[str]:
    return [line.strip() for line in provided_sources.splitlines() if line.strip()]


def _source_upload_to_entry(source_file: str, client: OpenAI) -> str:
    source_path = Path(source_file)
    parsed_source = parse_submission_file(source_path, openai_client=client)
    source_name = source_path.name
    return f"Uploaded Source ({source_name}):\n{parsed_source.extracted_text}"


def _build_sources(url_text: str, source_files: list[str] | None, client: OpenAI) -> list[str]:
    if len(url_text) > 1:
        source_urls = _normalize_url_sources(url_text)
        sources, status = build_sources_context(source_urls)
    else:
        sources = []
        status = []
    if len(source_files) > 0:
        sources_files, status_files = build_sources_context(source_files)
        try:
            sources = sources + sources_files
        except:
            sources.append(sources_files)
        try:
            status = status + status_files
        except:
            status.append(status_files)
    return sources, status


def grade_submission_ui(
    submission_files: list[str] | None,
    assignment_instructions: str,
    source_urls: str,
    source_files: list[str] | None,
    progress: gr.Progress = gr.Progress(),
):
    if not os.getenv("OPENAI_API_KEY"):
        yield "OPENAI_API_KEY is not configured."
        return

    if not submission_files:
        yield "Please upload at least one submission file."
        return

    if not assignment_instructions.strip():
        yield "Please provide assignment instructions."
        return

    client = OpenAI()
    temp_paths: list[Path] = []

    try:
        orchestrator = GradingOrchestrator(rubric_path=RUBRIC_PATH)
        sources, status = _build_sources(source_urls, source_files, client)
        status_strings = []
        for i in range(len(status)):
            if status[i]:
                status_strings.append(f"Source {i+1}: Good")
            else:
                status_strings.append(f"Source {i+1}: Error")
        graded_results: list[dict[str, Any]] = []

        total_submissions = len(submission_files)
        running_report_lines = [
            "# Grading Report",
            "",
            "## Progress",
            "",
            f"Processed 0/{total_submissions} submissions.",
            "",
            "## Results (streaming)",
            "",
        ]
        yield "\n".join(running_report_lines)
        for index, submission_file in enumerate(submission_files, start=1):
            progress((index - 1) / total_submissions, desc=f"Parsing {Path(submission_file).name}")
            original_path = Path(submission_file)
            suffix = original_path.suffix

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(original_path.read_bytes())
                temp_path = Path(temp.name)
                temp_paths.append(temp_path)

            parsed = parse_submission_file(temp_path, openai_client=client)
            progress((index - 1) / total_submissions, desc=f"Grading {original_path.name}")
            result = orchestrator.grade_submission(
                assignment_type=parsed.assignment_type,
                assignment_text=parsed.extracted_text,
                assignment_instructions=assignment_instructions,
                provided_sources=sources,
            )
            graded_item = {
                "student_name": original_path.stem,
                "submission_file": original_path.name,
                "result": result.model_dump(),
            }
            graded_results.append(graded_item)
            running_report_lines[4] = f"Processed {index}/{total_submissions} submissions."
            running_report_lines.extend(build_student_result_lines(graded_item))
            yield "\n".join(running_report_lines)

        report_data = {
            "assignment_instructions": assignment_instructions,
            "sources": sources,
            "source_proofs": status_strings,
            "results": graded_results,
        }
        progress(1, desc="Completed")
        yield build_markdown_report(report_data)
    except Exception as exc:  # surface human-readable errors to the UI
        yield f"Error while grading submission: {exc}"
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

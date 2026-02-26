from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from .agents import GradingOrchestrator
from .parsers import parse_submission_file
from .source_tools import fetch_sources_context


RUBRIC_PATH = Path("data/rubrics.json")


def grade_from_files(
    submission_file: Path,
    instructions_file: Path,
    sources_file: Path,
) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")

    instructions = instructions_file.read_text(encoding="utf-8")
    source_urls = [line.strip() for line in sources_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    parsed = parse_submission_file(submission_file, openai_client=OpenAI())
    orchestrator = GradingOrchestrator(rubric_path=RUBRIC_PATH)

    result = orchestrator.grade_submission(
        assignment_type=parsed.assignment_type,
        assignment_text=parsed.extracted_text,
        assignment_instructions=instructions,
        provided_sources=source_urls,
    )
    return result.model_dump()


def grade_multiple_from_files(
    submission_files: list[Path],
    instructions_file: Path,
    sources_file: Path,
) -> dict:
    if not submission_files:
        raise ValueError("At least one submission file is required")

    instructions = instructions_file.read_text(encoding="utf-8")
    source_urls = [line.strip() for line in sources_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_proofs = fetch_sources_context.invoke({"items": source_urls}) if source_urls else []

    def _grade_single(submission_path: Path) -> dict:
        single_result = grade_from_files(
            submission_file=submission_path,
            instructions_file=instructions_file,
            sources_file=sources_file,
        )
        student_name = submission_path.stem.replace("_", " ").replace("-", " ").strip() or submission_path.name
        return {
            "student_name": student_name,
            "submission_file": str(submission_path),
            "result": single_result,
        }

    max_workers = min(8, len(submission_files))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        graded_submissions = list(executor.map(_grade_single, submission_files))

    return {
        "assignment_instructions": instructions,
        "sources": source_urls,
        "source_proofs": source_proofs,
        "results": graded_submissions,
    }


def build_markdown_report(report_data: dict) -> str:
    lines = ["# Grading Report", "", "## Assignment instructions", "", report_data["assignment_instructions"], ""]

    lines.extend(["## Sources (with proof that the source was imported properly)", ""])
    if report_data["source_proofs"]:
        for source in report_data["source_proofs"]:
            source_type = source.get("type", "unknown")
            lines.append(f"- Source: `{source.get('source', 'unknown')}` ({source_type})")
            if source.get("ok"):
                proof_excerpt = (source.get("content") or "")[:250].replace("\n", " ").strip()
                lines.append(f"  - Imported: ✅")
                lines.append(f"  - Proof excerpt: `{proof_excerpt}`")
            else:
                lines.append(f"  - Imported: ❌")
                lines.append(f"  - Error: `{source.get('error', 'Unknown error')}`")
    else:
        lines.append("- No sources provided.")

    lines.extend(["", "## Results", ""])
    for graded in report_data["results"]:
        result = graded["result"]
        holistic = result["holistic"]
        lines.append(f"### Student name: {graded['student_name']}")
        lines.append(f"- Overall grade: {holistic['total_score']}/{holistic['max_total_score']}")
        lines.append("- Criteria:")
        for criterion in result["criteria_feedback"]:
            lines.append(f"  - name: {criterion['criterion_name']}")
            lines.append(f"    - grade: {criterion['score']}/{criterion['max_points']}")
            lines.append(f"    - feedback: {criterion['rationale']}")
            actionable_items = "; ".join(criterion.get("actionable_feedback", [])) or "None"
            lines.append(f"    - actionable items: {actionable_items}")
        lines.append(f"- Holistic summary: {holistic['summary']}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grading workflow from local files")
    parser.add_argument("--submission", action="append", type=Path, help="Path to one student submission (.docx/.pdf/.pptx/video). Repeat for multiple submissions.")
    parser.add_argument("--submissions-dir", type=Path, help="Directory containing student submissions to grade.")
    parser.add_argument("--instructions", required=True, type=Path, help="Path to txt file containing assignment instructions")
    parser.add_argument("--sources", required=True, type=Path, help="Path to txt file containing sources (URL, file path, or plain text) (one URL per line)")
    parser.add_argument("--output-markdown", type=Path, help="Write a markdown grading report to this path")
    args = parser.parse_args()

    submission_files: list[Path] = []
    if args.submission:
        submission_files.extend(args.submission)
    if args.submissions_dir:
        submission_files.extend(path for path in sorted(args.submissions_dir.iterdir()) if path.is_file())

    if not submission_files:
        parser.error("Provide at least one --submission or --submissions-dir")

    if len(submission_files) == 1 and not args.output_markdown:
        output = grade_from_files(
            submission_file=submission_files[0],
            instructions_file=args.instructions,
            sources_file=args.sources,
        )
        print(json.dumps(output, indent=2))
        return

    report_data = grade_multiple_from_files(
        submission_files=submission_files,
        instructions_file=args.instructions,
        sources_file=args.sources,
    )
    markdown_report = build_markdown_report(report_data)

    if args.output_markdown:
        args.output_markdown.write_text(markdown_report, encoding="utf-8")
        print(f"Markdown report written to {args.output_markdown}")
    else:
        print(markdown_report)


if __name__ == "__main__":
    main()

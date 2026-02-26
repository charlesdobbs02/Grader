from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from .agents import GradingOrchestrator
from .parsers import parse_submission_file


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grading workflow from local files")
    parser.add_argument("--submission", required=True, type=Path, help="Path to student submission (.docx/.pdf/.pptx/video)")
    parser.add_argument("--instructions", required=True, type=Path, help="Path to txt file containing assignment instructions")
    parser.add_argument("--sources", required=True, type=Path, help="Path to txt file containing sources (URL, file path, or plain text) (one URL per line)")
    args = parser.parse_args()

    output = grade_from_files(
        submission_file=args.submission,
        instructions_file=args.instructions,
        sources_file=args.sources,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

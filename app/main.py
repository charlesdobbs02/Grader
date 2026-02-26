from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import OpenAI

from .agents import GradingOrchestrator
from .models import GradeResponse
from .parsers import parse_submission_file

app = FastAPI(title="Agentic Assignment Grader")

RUBRIC_PATH = Path("data/rubrics.json")


def _require_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    return key


@app.post("/grade", response_model=GradeResponse)
async def grade_assignment(
    submission_file: UploadFile = File(...),
    assignment_instructions: str = Form(...),
    provided_sources: str = Form(
        ..., description="Newline-delimited list of allowed sources given with the assignment"
    ),
) -> GradeResponse:
    _require_openai_api_key()
    client = OpenAI()

    suffix = Path(submission_file.filename or "submission").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        payload = await submission_file.read()
        temp.write(payload)
        temp_path = Path(temp.name)

    try:
        parsed = parse_submission_file(temp_path, openai_client=client)
        orchestrator = GradingOrchestrator(rubric_path=RUBRIC_PATH)
        sources = [line.strip() for line in provided_sources.splitlines() if line.strip()]

        return orchestrator.grade_submission(
            assignment_type=parsed.assignment_type,
            assignment_text=parsed.extracted_text,
            assignment_instructions=assignment_instructions,
            provided_sources=sources,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink()

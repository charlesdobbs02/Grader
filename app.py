from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from grader.file_parser import parse_submission
from grader.pipeline import GradingOrchestrator, result_to_dict

app = FastAPI(title="Agentic Grader", version="0.1.0")
orchestrator = GradingOrchestrator()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/rubrics")
def rubrics() -> dict:
    return {"supported_assignment_types": orchestrator.rubric_store.supported_types()}


@app.post("/grade")
async def grade_submission(
    assignment_instructions: str = Form(...),
    submission_file: UploadFile = File(...),
    assignment_type: str | None = Form(default=None),
) -> dict:
    if not submission_file.filename:
        raise HTTPException(status_code=400, detail="submission_file requires a filename")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(submission_file.filename).suffix) as tmp:
        content = await submission_file.read()
        tmp.write(content)
        temp_path = Path(tmp.name)

    try:
        submission_text = parse_submission(temp_path)
        result = orchestrator.run(
            filename=submission_file.filename,
            submission_text=submission_text,
            assignment_instructions=assignment_instructions,
            assignment_type=assignment_type,
        )
        return result_to_dict(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

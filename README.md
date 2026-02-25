# Agentic Grader

Prototype app for grading student submissions with a LangGraph-orchestrated multi-agent workflow.

## Workflow

1. **Delegation agent** identifies assignment type and selects the correct rubric.
2. **Criterion grading agents** each score one criterion using rubric performance levels.
3. **Judging agent** reviews each criterion result and can request revisions.
4. **Total grading agent** combines criterion results into total score + holistic feedback.
5. **Judging agent** validates holistic feedback consistency and quality.

## Rubric Source

Rubrics are loaded from a permanent JSON artifact:

- `grader/rubrics/rubric.json`

Supported assignment types:
- `written assignment`
- `powerpoint`
- `video`

## API

### `GET /rubrics`
Returns supported rubric types loaded from the permanent rubric artifact.

### `POST /grade`
`multipart/form-data` fields:
- `assignment_instructions` (text)
- `submission_file` (file: txt/docx/pdf/pptx/video)
- `assignment_type` (optional override: `written assignment`, `essay`, `powerpoint`, `video`)

Response includes per-criterion scores/levels/feedback, total score, and holistic feedback.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app:app --reload
```

## Test

```bash
pytest
```

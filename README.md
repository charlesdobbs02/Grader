# Agentic Assignment Grader

FastAPI service that grades student assignments with a multi-agent LangGraph workflow.

## Features
- Accepts submissions as Word (`.docx`), PDF (`.pdf`), PowerPoint (`.pptx`), or video (`.mp4`, `.mov`, etc.).
- Delegation behavior is implemented by file parsing + assignment-type detection and rubric routing.
- Uses permanent rubric artifact in `data/rubrics.json`.
- Uses one grading agent per criterion and a judging agent loop for quality control.
- Uses a total grading agent for holistic summary and final score validation.

## API
`POST /grade` (multipart form)
- `submission_file`: uploaded student file
- `assignment_instructions`: text instructions
- `provided_sources`: newline-delimited sources (URL text or plain text)

## Run
```bash
export OPENAI_API_KEY=your_key_here
uvicorn app.main:app --reload
```

## Notes
- Video grading extracts audio and transcribes with OpenAI (`gpt-4o-mini-transcribe`).
- `.doc` and `.ppt` are treated as text fallback; convert to `.docx`/`.pptx` for best results.


## Gradio Frontend
Run a web UI for manual grading:
```bash
python -m app.gradio_ui
```
Then open `http://localhost:7860`.


## Source Access
The grader supports each `provided_sources` entry as HTTP(S) URL text or plain text. In the Gradio UI, source context can be supplied as URL text entries and/or uploaded source files.


## Workflow Tests (No Frontend)
Run backend workflow tests without using Gradio:
```bash
pytest tests/test_workflow_runner.py
```

Run the workflow manually using a submission file + instructions txt + sources txt:
```bash
python -m app.workflow_runner \
  --submission tests/fixtures/submission.docx \
  --instructions tests/fixtures/instructions.txt \
  --sources tests/fixtures/sources.txt
```

Run multiple submissions concurrently and write a markdown report:
```bash
python -m app.workflow_runner \
  --submission tests/fixtures/submission.docx \
  --submission path/to/another_student_submission.pdf \
  --instructions tests/fixtures/instructions.txt \
  --sources tests/fixtures/sources.txt \
  --output-markdown grading_report.md
```

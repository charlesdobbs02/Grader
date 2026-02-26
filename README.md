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
- `provided_sources`: newline-delimited source URLs

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


## Source URL Access
The grader fetches each URL in `provided_sources` (HTTP/HTTPS), extracts text content, and passes that context to grading agents.

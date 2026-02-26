from __future__ import annotations

from pathlib import Path

from app.models import (
    CriterionGrade,
    GradeResponse,
    HolisticGrade,
    ParseResult,
)
from app.workflow_runner import grade_from_files


def test_grade_from_files_uses_docx_and_text_inputs(monkeypatch, tmp_path: Path) -> None:
    submission_file = tmp_path / "submission.docx"
    submission_file.write_bytes(b"fake-docx-content")

    instructions_file = tmp_path / "instructions.txt"
    instructions_file.write_text("Write an evidence-based business memo.", encoding="utf-8")

    sources_file = tmp_path / "sources.txt"
    sources_file.write_text("https://example.com/source-a\nhttps://example.com/source-b\n", encoding="utf-8")

    captured: dict = {}

    def fake_parse_submission_file(path: Path, openai_client):
        captured["parsed_path"] = path
        return ParseResult(
            assignment_type="Written Assignment",
            extracted_text="Student memo content",
            parser_metadata={"kind": "fake"},
        )

    class FakeOrchestrator:
        def __init__(self, rubric_path: Path) -> None:
            captured["rubric_path"] = rubric_path

        def grade_submission(
            self,
            assignment_type: str,
            assignment_text: str,
            assignment_instructions: str,
            provided_sources: list[str],
        ) -> GradeResponse:
            captured["assignment_type"] = assignment_type
            captured["assignment_text"] = assignment_text
            captured["assignment_instructions"] = assignment_instructions
            captured["provided_sources"] = provided_sources
            return GradeResponse(
                assignment_type=assignment_type,
                criteria_feedback=[
                    CriterionGrade(
                        criterion_name="Content & Analysis",
                        score=30,
                        max_points=40,
                        performance_level="Accomplished",
                        rationale="Reasonable analysis with some gaps.",
                        actionable_feedback=["Deepen model justification"],
                    )
                ],
                holistic=HolisticGrade(
                    total_score=30,
                    max_total_score=100,
                    summary="Solid start with room for deeper evidence integration.",
                    strengths=["Clear structure"],
                    areas_to_improve=["Use more source evidence"],
                ),
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("app.workflow_runner.parse_submission_file", fake_parse_submission_file)
    monkeypatch.setattr("app.workflow_runner.GradingOrchestrator", FakeOrchestrator)

    result = grade_from_files(
        submission_file=submission_file,
        instructions_file=instructions_file,
        sources_file=sources_file,
    )

    assert captured["parsed_path"] == submission_file
    assert captured["assignment_type"] == "Written Assignment"
    assert captured["assignment_text"] == "Student memo content"
    assert captured["assignment_instructions"] == "Write an evidence-based business memo."
    assert captured["provided_sources"] == [
        "https://example.com/source-a",
        "https://example.com/source-b",
    ]
    assert result["assignment_type"] == "Written Assignment"
    assert result["holistic"]["total_score"] == 30


def test_grade_from_files_requires_openai_key(tmp_path: Path, monkeypatch) -> None:
    submission_file = tmp_path / "submission.docx"
    instructions_file = tmp_path / "instructions.txt"
    sources_file = tmp_path / "sources.txt"
    submission_file.write_bytes(b"x")
    instructions_file.write_text("instructions", encoding="utf-8")
    sources_file.write_text("https://example.com", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        grade_from_files(submission_file, instructions_file, sources_file)
        raise AssertionError("Expected RuntimeError for missing OPENAI_API_KEY")
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)

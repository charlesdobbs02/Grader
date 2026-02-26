from __future__ import annotations

from pathlib import Path

from app.workflow_runner import build_markdown_report, grade_multiple_from_files


def test_grade_multiple_from_files_aggregates_results(monkeypatch, tmp_path: Path) -> None:
    instructions_file = tmp_path / "instructions.txt"
    instructions_file.write_text("Follow assignment directions.", encoding="utf-8")

    sources_file = tmp_path / "sources.txt"
    sources_file.write_text("https://example.com/a\n", encoding="utf-8")

    submission_a = tmp_path / "Jane_Doe.docx"
    submission_b = tmp_path / "john-smith.pdf"
    submission_a.write_bytes(b"a")
    submission_b.write_bytes(b"b")

    def fake_grade_from_files(submission_file: Path, instructions_file: Path, sources_file: Path) -> dict:
        return {
            "assignment_type": "Written Assignment",
            "criteria_feedback": [
                {
                    "criterion_name": "Accuracy",
                    "score": 8,
                    "max_points": 10,
                    "rationale": "Mostly correct.",
                    "actionable_feedback": ["Add one more citation"],
                }
            ],
            "holistic": {
                "total_score": 8,
                "max_total_score": 10,
                "summary": f"Summary for {submission_file.name}",
            },
        }

    monkeypatch.setattr("app.workflow_runner.grade_from_files", fake_grade_from_files)
    monkeypatch.setattr(
        "app.workflow_runner.fetch_sources_context",
        type("FakeTool", (), {"invoke": staticmethod(lambda payload: [{"source": payload["items"][0], "type": "url", "ok": True, "content": "Imported source text"}] )}),
    )

    result = grade_multiple_from_files(
        submission_files=[submission_a, submission_b],
        instructions_file=instructions_file,
        sources_file=sources_file,
    )

    assert result["assignment_instructions"] == "Follow assignment directions."
    assert result["sources"] == ["https://example.com/a"]
    assert len(result["results"]) == 2
    assert result["results"][0]["student_name"] == "Jane Doe"
    assert result["results"][1]["student_name"] == "john smith"
    assert result["source_proofs"][0]["ok"] is True


def test_build_markdown_report_matches_requested_sections() -> None:
    report = {
        "assignment_instructions": "Do the task.",
        "source_proofs": [
            {"source": "https://example.com", "type": "url", "ok": True, "content": "Evidence pulled from source"}
        ],
        "results": [
            {
                "student_name": "Jane Doe",
                "result": {
                    "criteria_feedback": [
                        {
                            "criterion_name": "Structure",
                            "score": 9,
                            "max_points": 10,
                            "rationale": "Well organized.",
                            "actionable_feedback": ["Tighten conclusion"],
                        }
                    ],
                    "holistic": {"total_score": 9, "max_total_score": 10, "summary": "Great work."},
                },
            }
        ],
    }

    markdown = build_markdown_report(report)

    assert "## Assignment instructions" in markdown
    assert "## Sources (with proof that the source was imported properly)" in markdown
    assert "## Results" in markdown
    assert "### Student name: Jane Doe" in markdown
    assert "- Overall grade: 9/10" in markdown
    assert "- Criteria:" in markdown
    assert "- Holistic summary: Great work." in markdown

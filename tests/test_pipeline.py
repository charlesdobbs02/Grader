from grader.pipeline import GradingOrchestrator


def test_orchestrator_scores_and_totals():
    orchestrator = GradingOrchestrator(max_revisions=1)
    result = orchestrator.run(
        filename="essay.txt",
        submission_text=(
            "This written assignment includes evidence, analysis, data, models, recommendations, "
            "critical evaluation, and a clear conclusion."
        ),
        assignment_instructions="Write a written assignment with APA formatting and evidence.",
    )

    assert result.assignment_type == "written assignment"
    assert len(result.criteria_results) == 5
    assert result.total_score == sum(item.score for item in result.criteria_results)
    assert result.max_score == sum(item.max_points for item in result.criteria_results)
    assert "Overall performance" in result.holistic_feedback


def test_explicit_assignment_type_overrides_detection_with_alias():
    orchestrator = GradingOrchestrator(max_revisions=0)
    result = orchestrator.run(
        filename="submission.txt",
        submission_text="content",
        assignment_instructions="some instructions",
        assignment_type="essay",
    )

    assert result.assignment_type == "written assignment"
    assert len(result.criteria_results) == 5

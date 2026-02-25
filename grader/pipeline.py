from __future__ import annotations

from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from grader.models import Criterion, CriterionResult, GradeResult, PerformanceLevel
from grader.rubric_store import RubricStore


class DelegationAgent:
    def __init__(self, rubric_store: RubricStore) -> None:
        self._rubric_store = rubric_store

    def detect_assignment_type(self, filename: str, assignment_instructions: str, explicit_type: str | None = None) -> str:
        if explicit_type:
            return self._rubric_store.normalize_type(explicit_type)

        text = f"{filename} {assignment_instructions}".lower()
        if any(token in text for token in ["video", ".mp4", "recording"]):
            return "video"
        if any(token in text for token in ["slides", "powerpoint", ".ppt", ".pptx", "deck"]):
            return "powerpoint"
        return "written assignment"

    def pick_rubric(self, assignment_type: str) -> List[Criterion]:
        return self._rubric_store.get_rubric(assignment_type)


class GradingAgent:
    def __init__(self, criterion: Criterion) -> None:
        self.criterion = criterion

    def _select_level(self, submission_text: str, instructions: str, judge_feedback: str | None = None) -> PerformanceLevel:
        text = f"{submission_text}\n{instructions}".lower()
        strength_signals = [
            "analysis",
            "evidence",
            "data",
            "model",
            "recommend",
            "conclusion",
            "critical",
            "organized",
            "professional",
        ]
        score = sum(1 for token in strength_signals if token in text)
        if judge_feedback:
            score += 2

        if score >= 6:
            return self.criterion.performance_levels[0]
        if score >= 4:
            return self.criterion.performance_levels[min(1, len(self.criterion.performance_levels) - 1)]
        if score >= 2:
            return self.criterion.performance_levels[min(2, len(self.criterion.performance_levels) - 1)]
        return self.criterion.performance_levels[-1]

    def grade(self, submission_text: str, instructions: str, judge_feedback: str | None = None) -> CriterionResult:
        level = self._select_level(submission_text, instructions, judge_feedback)
        feedback = (
            f"{self.criterion.name}: {level.name} ({level.points}/{self.criterion.max_points}). "
            f"Rationale: {level.grading_criteria}"
        )
        return CriterionResult(
            criterion=self.criterion.name,
            level=level.name,
            score=level.points,
            max_points=self.criterion.max_points,
            feedback=feedback,
        )


class JudgingAgent:
    def review_criterion(self, result: CriterionResult) -> tuple[bool, str]:
        quality_ok = len(result.feedback.split()) >= 16 and result.score <= result.max_points
        if quality_ok:
            return True, "Accepted"
        return False, "Criterion feedback is too short or score exceeds maximum. Expand justification and align with rubric level."

    def review_holistic(self, holistic_feedback: str, criteria_results: List[CriterionResult]) -> tuple[bool, str]:
        mentions_all = all(result.criterion.lower() in holistic_feedback.lower() for result in criteria_results)
        if mentions_all and len(holistic_feedback.split()) >= 30:
            return True, "Accepted"
        return False, "Holistic feedback must mention every criterion and include stronger synthesis."


class TotalGradingAgent:
    def combine(self, assignment_type: str, source_summary: str, criteria_results: List[CriterionResult]) -> GradeResult:
        total = sum(c.score for c in criteria_results)
        max_score = sum(c.max_points for c in criteria_results)
        bullets = " ".join(
            f"{c.criterion}: {c.score}/{c.max_points} ({c.level})." for c in criteria_results
        )
        holistic = (
            f"Overall performance for {assignment_type}: {bullets} "
            "This synthesis reflects strengths and growth areas across the full rubric. "
            "Prioritize the lowest-scoring criteria first to maximize improvement in future submissions."
        )
        return GradeResult(
            assignment_type=assignment_type,
            criteria_results=criteria_results,
            total_score=total,
            max_score=max_score,
            holistic_feedback=holistic,
            source_summary=source_summary[:500],
        )


class PipelineState(TypedDict, total=False):
    filename: str
    submission_text: str
    assignment_instructions: str
    assignment_type: str | None
    resolved_type: str
    rubric: List[Criterion]
    criterion_index: int
    criterion_results: List[CriterionResult]
    current_result: CriterionResult
    judge_feedback: str | None
    criterion_accepted: bool
    criterion_revision: int
    final_result: GradeResult
    holistic_accepted: bool
    holistic_revision: int


class GradingOrchestrator:
    def __init__(self, rubric_store: RubricStore | None = None, max_revisions: int = 2) -> None:
        self.rubric_store = rubric_store or RubricStore()
        self.delegation_agent = DelegationAgent(self.rubric_store)
        self.judging_agent = JudgingAgent()
        self.total_grading_agent = TotalGradingAgent()
        self.max_revisions = max_revisions
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("delegate", self._delegate)
        graph.add_node("grade_criterion", self._grade_criterion)
        graph.add_node("judge_criterion", self._judge_criterion)
        graph.add_node("advance_criterion", self._advance_criterion)
        graph.add_node("combine_total", self._combine_total)
        graph.add_node("judge_holistic", self._judge_holistic)
        graph.add_node("revise_holistic", self._revise_holistic)

        graph.set_entry_point("delegate")
        graph.add_edge("delegate", "grade_criterion")
        graph.add_edge("grade_criterion", "judge_criterion")
        graph.add_conditional_edges("judge_criterion", self._route_criterion)
        graph.add_conditional_edges("advance_criterion", self._route_after_advance)
        graph.add_edge("combine_total", "judge_holistic")
        graph.add_conditional_edges("judge_holistic", self._route_holistic)
        graph.add_edge("revise_holistic", "judge_holistic")

        return graph.compile()

    def _delegate(self, state: PipelineState) -> PipelineState:
        resolved_type = self.delegation_agent.detect_assignment_type(
            filename=state["filename"],
            assignment_instructions=state["assignment_instructions"],
            explicit_type=state.get("assignment_type"),
        )
        return {
            "resolved_type": resolved_type,
            "rubric": self.delegation_agent.pick_rubric(resolved_type),
            "criterion_index": 0,
            "criterion_results": [],
            "criterion_revision": 0,
            "holistic_revision": 0,
            "judge_feedback": None,
        }

    def _grade_criterion(self, state: PipelineState) -> PipelineState:
        criterion = state["rubric"][state["criterion_index"]]
        result = GradingAgent(criterion).grade(
            submission_text=state["submission_text"],
            instructions=state["assignment_instructions"],
            judge_feedback=state.get("judge_feedback"),
        )
        result.revisions = state.get("criterion_revision", 0)
        return {"current_result": result}

    def _judge_criterion(self, state: PipelineState) -> PipelineState:
        accepted, feedback = self.judging_agent.review_criterion(state["current_result"])
        return {
            "criterion_accepted": accepted,
            "criterion_revision": state.get("criterion_revision", 0) + (0 if accepted else 1),
            "judge_feedback": None if accepted else feedback,
        }

    def _route_criterion(self, state: PipelineState) -> str:
        if state["criterion_accepted"] or state.get("criterion_revision", 0) > self.max_revisions:
            return "advance_criterion"
        return "grade_criterion"

    def _advance_criterion(self, state: PipelineState) -> PipelineState:
        results = list(state["criterion_results"])
        results.append(state["current_result"])
        return {
            "criterion_results": results,
            "criterion_index": state["criterion_index"] + 1,
            "criterion_revision": 0,
            "judge_feedback": None,
        }

    def _route_after_advance(self, state: PipelineState) -> str:
        if state["criterion_index"] < len(state["rubric"]):
            return "grade_criterion"
        return "combine_total"

    def _combine_total(self, state: PipelineState) -> PipelineState:
        result = self.total_grading_agent.combine(
            assignment_type=state["resolved_type"],
            source_summary=state["submission_text"],
            criteria_results=state["criterion_results"],
        )
        return {"final_result": result}

    def _judge_holistic(self, state: PipelineState) -> PipelineState:
        accepted, feedback = self.judging_agent.review_holistic(
            state["final_result"].holistic_feedback,
            state["final_result"].criteria_results,
        )
        return {"holistic_accepted": accepted, "judge_feedback": feedback if not accepted else None}

    def _route_holistic(self, state: PipelineState) -> str:
        if state["holistic_accepted"] or state.get("holistic_revision", 0) > self.max_revisions:
            return END
        return "revise_holistic"

    def _revise_holistic(self, state: PipelineState) -> PipelineState:
        revised = state["final_result"]
        revised.holistic_feedback = f"{revised.holistic_feedback} Revision note: {state.get('judge_feedback', '')}"
        return {"final_result": revised, "holistic_revision": state.get("holistic_revision", 0) + 1}

    def run(self, filename: str, submission_text: str, assignment_instructions: str, assignment_type: str | None = None) -> GradeResult:
        state = self._graph.invoke(
            {
                "filename": filename,
                "submission_text": submission_text,
                "assignment_instructions": assignment_instructions,
                "assignment_type": assignment_type,
            }
        )
        return state["final_result"]


def result_to_dict(result: GradeResult) -> dict:
    return result.model_dump()

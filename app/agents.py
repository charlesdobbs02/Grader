from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .models import (
    Criterion,
    CriterionGrade,
    GradeResponse,
    HolisticGrade,
    JudgeFeedback,
    Rubric,
)
from .source_tools import build_sources_context


class WorkflowState(TypedDict, total=False):
    assignment_type: str
    assignment_text: str
    assignment_instructions: str
    provided_sources: list[str]
    provided_sources_context: str
    rubric: Rubric
    criteria_results: list[CriterionGrade]
    criterion_index: int
    current_criterion: Criterion
    candidate_grade: CriterionGrade
    criterion_revision_feedback: str | None
    criterion_revision_count: int
    holistic_candidate: HolisticGrade
    holistic_revision_count: int
    holistic_accepted: bool


class GradingOrchestrator:
    def __init__(self, rubric_path: Path, model: str = "gpt-4o") -> None:
        self.model = ChatOpenAI(model=model, temperature=0)
        self.rubrics = self._load_rubrics(rubric_path)
        self.graph = self._build_graph()

    @staticmethod
    def _load_rubrics(path: Path) -> dict[str, Rubric]:
        data = json.loads(path.read_text())
        rubrics = [Rubric.model_validate(item) for item in data]
        return {rubric.Assignment_Type: rubric for rubric in rubrics}

    def _grade_criterion(
        self,
        criterion: Criterion,
        assignment_text: str,
        assignment_instructions: str,
        provided_sources: list[str],
        provided_sources_context: str,
        revision_feedback: str | None = None,
    ) -> CriterionGrade:
        grader_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a strict grading agent. Grade only one criterion from the rubric. "
                    "Use the rubric levels exactly. Output concise, specific, evidence-based feedback.",
                ),
                (
                    "human",
                    "Criterion: {criterion}\n"
                    "Assignment Instructions:\n{instructions}\n\n"
                    "Allowed Source URLs:\n{sources}\n\n"
                    "Fetched Source Content:\n{source_context}\n\n"
                    "Submission Content:\n{submission}\n\n"
                    "Judge Revision Feedback (if any): {revision_feedback}\n"
                    "Return score <= max points and actionable feedback.",
                ),
            ]
        )
        structured = self.model.with_structured_output(CriterionGrade)
        return structured.invoke(
            grader_prompt.format_messages(
                criterion=criterion.model_dump_json(indent=2),
                instructions=assignment_instructions,
                sources="\n".join(provided_sources) if provided_sources else "None provided",
                source_context=provided_sources_context,
                submission=assignment_text,
                revision_feedback=revision_feedback or "None",
            )
        )

    def _judge_criterion(
        self,
        criterion: Criterion,
        criterion_grade: CriterionGrade,
        assignment_text: str,
        assignment_instructions: str,
    ) -> JudgeFeedback:
        judge_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a judging agent validating grading quality. "
                    "Reject grades that are inconsistent with rubric or weakly justified.",
                ),
                (
                    "human",
                    "Criterion rubric:\n{criterion}\n\n"
                    "Submission:\n{submission}\n\n"
                    "Instructions:\n{instructions}\n\n"
                    "Grading output to evaluate:\n{grade}",
                ),
            ]
        )
        structured = self.model.with_structured_output(JudgeFeedback)
        return structured.invoke(
            judge_prompt.format_messages(
                criterion=criterion.model_dump_json(indent=2),
                submission=assignment_text,
                instructions=assignment_instructions,
                grade=criterion_grade.model_dump_json(indent=2),
            )
        )

    def _finalize_holistic(
        self,
        assignment_type: str,
        criteria_grades: list[CriterionGrade],
        assignment_text: str,
        assignment_instructions: str,
    ) -> HolisticGrade:
        total_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a total grading agent. Summarize all criteria results holistically and consistently. "
                    "Total score must equal sum of criteria scores.",
                ),
                (
                    "human",
                    "Assignment type: {assignment_type}\n"
                    "Instructions:\n{instructions}\n\n"
                    "Submission excerpt:\n{submission}\n\n"
                    "Criterion results:\n{grades}",
                ),
            ]
        )
        structured = self.model.with_structured_output(HolisticGrade)
        return structured.invoke(
            total_prompt.format_messages(
                assignment_type=assignment_type,
                instructions=assignment_instructions,
                submission=assignment_text[:8000],
                grades=json.dumps([g.model_dump() for g in criteria_grades], indent=2),
            )
        )

    def _judge_holistic(
        self,
        assignment_type: str,
        criteria_grades: list[CriterionGrade],
        holistic: HolisticGrade,
    ) -> JudgeFeedback:
        judge_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a judging agent validating final holistic feedback for contradictions and quality.",
                ),
                (
                    "human",
                    "Assignment type: {assignment_type}\n"
                    "Criteria grades:\n{criteria}\n\n"
                    "Holistic output:\n{holistic}",
                ),
            ]
        )
        structured = self.model.with_structured_output(JudgeFeedback)
        return structured.invoke(
            judge_prompt.format_messages(
                assignment_type=assignment_type,
                criteria=json.dumps([g.model_dump() for g in criteria_grades], indent=2),
                holistic=holistic.model_dump_json(indent=2),
            )
        )

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("initialize", self._initialize_state)
        graph.add_node("select_criterion", self._select_criterion)
        graph.add_node("grade_criterion", self._grade_criterion_node)
        graph.add_node("judge_criterion", self._judge_criterion_node)
        graph.add_node("build_holistic", self._build_holistic_node)
        graph.add_node("judge_holistic", self._judge_holistic_node)

        graph.set_entry_point("initialize")
        graph.add_edge("initialize", "select_criterion")
        graph.add_conditional_edges(
            "select_criterion",
            self._route_after_select_criterion,
            {"grade_criterion": "grade_criterion", "build_holistic": "build_holistic"},
        )
        graph.add_edge("grade_criterion", "judge_criterion")
        graph.add_conditional_edges(
            "judge_criterion",
            self._route_after_judge_criterion,
            {"grade_criterion": "grade_criterion", "select_criterion": "select_criterion"},
        )
        graph.add_edge("build_holistic", "judge_holistic")
        graph.add_conditional_edges(
            "judge_holistic",
            self._route_after_judge_holistic,
            {"build_holistic": "build_holistic", "end": END},
        )
        return graph.compile()

    def _initialize_state(self, state: WorkflowState) -> WorkflowState:
        return {
            "rubric": self.rubrics[state["assignment_type"]],
            "criteria_results": [],
            "criterion_index": 0,
            "criterion_revision_feedback": None,
            "criterion_revision_count": 0,
            "holistic_revision_count": 0,
            "holistic_accepted": False,
        }

    def _select_criterion(self, state: WorkflowState) -> WorkflowState:
        rubric = state["rubric"]
        index = state["criterion_index"]
        if index >= len(rubric.Criteria):
            return {}
        return {
            "current_criterion": rubric.Criteria[index],
            "criterion_revision_feedback": None,
            "criterion_revision_count": 0,
        }

    def _route_after_select_criterion(self, state: WorkflowState) -> str:
        if state["criterion_index"] >= len(state["rubric"].Criteria):
            return "build_holistic"
        return "grade_criterion"

    def _grade_criterion_node(self, state: WorkflowState) -> WorkflowState:
        candidate = self._grade_criterion(
            criterion=state["current_criterion"],
            assignment_text=state["assignment_text"],
            assignment_instructions=state["assignment_instructions"],
            provided_sources=state["provided_sources"],
            provided_sources_context=state["provided_sources_context"],
            revision_feedback=state.get("criterion_revision_feedback"),
        )
        candidate.max_points = state["current_criterion"].Max_Points
        candidate.score = min(candidate.score, candidate.max_points)
        return {"candidate_grade": candidate}

    def _judge_criterion_node(self, state: WorkflowState) -> WorkflowState:
        judge = self._judge_criterion(
            criterion=state["current_criterion"],
            criterion_grade=state["candidate_grade"],
            assignment_text=state["assignment_text"],
            assignment_instructions=state["assignment_instructions"],
        )
        revision_count = state.get("criterion_revision_count", 0)
        if judge.acceptable or revision_count >= 2:
            updated_results = list(state.get("criteria_results", []))
            updated_results.append(state["candidate_grade"])
            return {
                "criteria_results": updated_results,
                "criterion_index": state["criterion_index"] + 1,
                "criterion_revision_feedback": None,
                "criterion_revision_count": 0,
            }

        return {
            "criterion_revision_feedback": judge.feedback_for_revision,
            "criterion_revision_count": revision_count + 1,
        }

    def _route_after_judge_criterion(self, state: WorkflowState) -> str:
        if state.get("criterion_revision_feedback"):
            return "grade_criterion"
        return "select_criterion"

    def _build_holistic_node(self, state: WorkflowState) -> WorkflowState:
        holistic = self._finalize_holistic(
            assignment_type=state["assignment_type"],
            criteria_grades=state["criteria_results"],
            assignment_text=state["assignment_text"],
            assignment_instructions=state["assignment_instructions"],
        )
        holistic.total_score = sum(item.score for item in state["criteria_results"])
        holistic.max_total_score = sum(item.max_points for item in state["criteria_results"])
        return {"holistic_candidate": holistic}

    def _judge_holistic_node(self, state: WorkflowState) -> WorkflowState:
        judge = self._judge_holistic(
            assignment_type=state["assignment_type"],
            criteria_grades=state["criteria_results"],
            holistic=state["holistic_candidate"],
        )
        revision_count = state.get("holistic_revision_count", 0)
        if judge.acceptable or revision_count >= 2:
            return {"holistic_accepted": True}
        return {"holistic_revision_count": revision_count + 1, "holistic_accepted": False}

    def _route_after_judge_holistic(self, state: WorkflowState) -> str:
        if state.get("holistic_accepted"):
            return "end"
        return "build_holistic"

    def grade_submission(
        self,
        assignment_type: str,
        assignment_text: str,
        assignment_instructions: str,
        provided_sources: list[str],
    ) -> GradeResponse:
        source_context = build_sources_context(provided_sources) if provided_sources else "None provided."

        final_state = self.graph.invoke(
            {
                "assignment_type": assignment_type,
                "assignment_text": assignment_text,
                "assignment_instructions": assignment_instructions,
                "provided_sources": provided_sources,
                "provided_sources_context": source_context,
            }
        )

        return GradeResponse(
            assignment_type=assignment_type,
            criteria_feedback=final_state["criteria_results"],
            holistic=final_state["holistic_candidate"],
        )

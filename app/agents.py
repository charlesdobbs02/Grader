from __future__ import annotations
import time
import json
from pathlib import Path
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from models import (
    Criterion,
    CriterionGrade,
    GradeResponse,
    HolisticGrade,
    JudgeFeedback,
    Rubric,
)
from source_tools import build_sources_context


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
    def __init__(self, rubric_path: Path, model: str = "gpt-5.4") -> None:
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
                    """
                    You are a rubric-scoring assistant evaluating a student paper against exactly one rubric criterion.

                    You will receive:
                    1. A single rubric criterion, including scoring levels or performance descriptors
                    2. A student paper

                    Your task is to assign the most appropriate score for that one criterion only.

                    Evaluation procedure:
                    - Read the criterion first and identify what it actually measures.
                    - Read the student paper and collect only evidence relevant to that criterion.
                    - Match the paper to the best-fitting rubric level.
                    - Justify the score with direct, paper-based evidence.
                    - Ignore all unrelated qualities.
                    - Pay very close attention to the assignment instructions, the instructions supercede the criterion if any aspect of the criterion does not apply
                    - If the assignment instructions do not call for charts, figures, or forecast models, do not grade based on these.

                    Scoring principles:
                    - Use only the supplied rubric language and score scale.
                    - Do not use outside standards.
                    - Do not guess at missing content.
                    - Do not reward intent; reward demonstrated performance.
                    - When evidence is mixed, select the level that is fully supported by the paper.
                    - When deciding between two adjacent levels, prefer the higher level unless the lower level is clearly met.

                    Feedback principles:
                    - Be concise, specific, and rubric-aligned.
                    - Reference concrete features of the student paper.
                    - Avoid generic comments such as “good job” or “needs more detail” unless tied directly to the criterion.
                    - Suggestions for improvement must address only this criterion.
                    - Provide specific examples from the submission text at all times.
                    - Since you are only receiving plain text, APA formatting judgement will ONLY indclude errors in citations, DO NOT provide other APA feedback.
                    - Give feedback in second person ONLY.
                    """,
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
        max_retries = 5
        retries = 0
        cont = True
        while cont:
            try:
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
            except:
                time.sleep(10)
                retries += 1
                if retries >= max_retries:
                    cont = False
                continue

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
                    """You are a rubric-grade auditor reviewing whether a grading agent correctly evaluated a student paper on exactly one rubric criterion.

                    Inputs:
                    - rubric_criterion
                    - rubric_levels
                    - student_paper
                    - grader_score
                    - grader_reasoning

                    Your task is to assess the grader’s evaluation quality, not merely to regrade the paper from scratch.

                    Review standards:
                    - Use the rubric as the authoritative standard
                    - Verify that the grader stayed focused on the single criterion
                    - Verify that the grader’s reasoning is grounded in evidence from the paper
                    - Check whether the grader’s score matches the rubric descriptors
                    - Identify unsupported claims, criterion drift, inconsistency, or overreach
                    - Decide whether the grader’s judgment should be upheld, revised, or rejected

                    Decision rules:
                    - "true" if the grader’s score and rationale are well supported
                    - "false" if the grader shows partial validity but has reasoning gaps or mild score misalignment or if the grader’s score is not supported by the rubric and paper
                    """,
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
        max_retries = 5
        retries = 0
        cont = True
        while cont:
            try:
                return structured.invoke(
                    judge_prompt.format_messages(
                        criterion=criterion.model_dump_json(indent=2),
                        submission=assignment_text,
                        instructions=assignment_instructions,
                        grade=criterion_grade.model_dump_json(indent=2),
                    )
                )
            except:
                time.sleep(10)
                retries += 1
                if retries >= max_retries:
                    cont = False
                continue

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
                    """You are a grading feedback aggregation agent. Your role is to synthesize outputs from multiple criterion-level grading agents into one coherent summary.

                    Inputs:
                    - criteria_feedback: an array of objects, each containing:
                    - criterion
                    - assigned_score_or_level
                    - reasoning
                    - evidence_from_paper
                    - improvement_for_this_criterion
                    - confidence
                    - optional_rubric
                    - optional_student_paper

                    Your task:
                    - Summarize the feedback across criteria without regrading
                    - Preserve the meaning of each criterion judgment
                    - Identify recurring strengths and recurring areas for improvement
                    - Produce a concise, actionable, and student-friendly synthesis

                    Rules:
                    - Do not assign new scores
                    - Do not alter criterion-level scores
                    - Do not invent evidence
                    - Do not independently evaluate criteria unless explicitly instructed
                    - Use criterion outputs as the primary source of truth
                    - If different criterion agents conflict, note the inconsistency briefly rather than guessing
                    """,
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
        max_retries = 5
        retries = 0
        cont = True
        while cont:
            try:
                return structured.invoke(
                    total_prompt.format_messages(
                        assignment_type=assignment_type,
                        instructions=assignment_instructions,
                        submission=assignment_text,
                        grades=json.dumps([g.model_dump() for g in criteria_grades], indent=2),
                    )
                )
            except:
                time.sleep(10)
                retries += 1
                if retries >= max_retries:
                    cont = False
                continue

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
                    """You are a summary-quality auditor reviewing whether a feedback synthesis agent correctly summarized multiple criterion-level grading outputs.

                    Inputs:
                    - criteria_feedback: array of criterion-level grading outputs
                    - summary_output: the synthesis produced by the summary agent
                    - optional_rubric
                    - optional_student_paper

                    Your task:
                    - Evaluate whether the summary_output faithfully represents the criteria_feedback
                    - Check for omissions, distortions, invented claims, imbalance, and weak actionability
                    - Determine whether the summary should be upheld, revised, or rejected

                    Decision rules:
                    - "true" if the summary is materially accurate, balanced, and useful
                    - "false" if the summary is mostly accurate but has notable omissions, imprecision, or weak actionability or if the summary materially misrepresents the criterion-level feedback
                    """,
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
        max_retries = 5
        retries = 0
        cont = True
        while cont:
            try:
                return structured.invoke(
                    judge_prompt.format_messages(
                        assignment_type=assignment_type,
                        criteria=json.dumps([g.model_dump() for g in criteria_grades], indent=2),
                        holistic=holistic.model_dump_json(indent=2),
                    )
                )
            except:
                time.sleep(10)
                retries += 1
                if retries >= max_retries:
                    cont = False
                continue

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
        provided_sources: str,
    ) -> GradeResponse:
        source_context = provided_sources

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

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class PerformanceLevel(BaseModel):
    Name: str
    Points: float
    Grading_Criteria: str


class Criterion(BaseModel):
    Name: str
    Max_Points: float
    Performance_Levels: List[PerformanceLevel]


class Rubric(BaseModel):
    Assignment_Type: Literal["Written Assignment", "PowerPoint", "Video"]
    Criteria: List[Criterion]


class ParseResult(BaseModel):
    assignment_type: Literal["Written Assignment", "PowerPoint", "Video"]
    extracted_text: str
    parser_metadata: dict = Field(default_factory=dict)


class CriterionGrade(BaseModel):
    criterion_name: str
    score: float
    max_points: float
    performance_level: str
    rationale: str
    actionable_feedback: List[str]


class JudgeFeedback(BaseModel):
    acceptable: bool
    feedback_for_revision: str


class HolisticGrade(BaseModel):
    total_score: float
    max_total_score: float
    summary: str
    strengths: List[str]
    areas_to_improve: List[str]


class GradeResponse(BaseModel):
    assignment_type: str
    criteria_feedback: List[CriterionGrade]
    holistic: HolisticGrade

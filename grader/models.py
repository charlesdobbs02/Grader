from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PerformanceLevel(BaseModel):
    name: str
    points: float
    grading_criteria: str


class Criterion(BaseModel):
    name: str
    max_points: float
    performance_levels: List[PerformanceLevel]


class CriterionResult(BaseModel):
    criterion: str
    level: str
    score: float
    max_points: float
    feedback: str
    revisions: int = 0


class GradeResult(BaseModel):
    assignment_type: str
    criteria_results: List[CriterionResult] = Field(default_factory=list)
    total_score: float = 0.0
    max_score: float = 0.0
    holistic_feedback: str = ""
    source_summary: str = ""

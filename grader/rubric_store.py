from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from grader.models import Criterion, PerformanceLevel


ALIASES = {
    "essay": "written assignment",
    "written": "written assignment",
    "written assignment": "written assignment",
    "powerpoint": "powerpoint",
    "ppt": "powerpoint",
    "presentation": "powerpoint",
    "video": "video",
}


class RubricStore:
    def __init__(self, rubric_path: Path | None = None) -> None:
        self._rubric_path = rubric_path or Path(__file__).parent / "rubrics" / "rubric.json"
        self._rubrics = self._load_rubrics()

    def _load_rubrics(self) -> Dict[str, List[Criterion]]:
        payload = json.loads(self._rubric_path.read_text())
        rubrics: Dict[str, List[Criterion]] = {}

        for assignment in payload:
            assignment_type = assignment["Assignment_Type"].strip().lower()
            rubrics[assignment_type] = []
            for criterion in assignment["Criteria"]:
                levels = [
                    PerformanceLevel(
                        name=level["Name"],
                        points=float(level["Points"]),
                        grading_criteria=level["Grading_Criteria"],
                    )
                    for level in criterion["Performance_Levels"]
                ]
                levels.sort(key=lambda level: level.points, reverse=True)
                rubrics[assignment_type].append(
                    Criterion(
                        name=criterion["Name"],
                        max_points=float(criterion["Max_Points"]),
                        performance_levels=levels,
                    )
                )
        return rubrics

    def normalize_type(self, assignment_type: str) -> str:
        normalized = assignment_type.strip().lower()
        return ALIASES.get(normalized, normalized)

    def get_rubric(self, assignment_type: str) -> List[Criterion]:
        normalized = self.normalize_type(assignment_type)
        if normalized not in self._rubrics:
            valid = ", ".join(sorted(self._rubrics))
            raise ValueError(f"Unknown assignment type '{assignment_type}'. Valid: {valid}")
        return self._rubrics[normalized]

    def supported_types(self) -> List[str]:
        return sorted(self._rubrics.keys())

from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_rubrics_endpoint():
    response = client.get("/rubrics")
    assert response.status_code == 200
    assert set(response.json()["supported_assignment_types"]) == {"written assignment", "powerpoint", "video"}


def test_grade_endpoint_text_file():
    files = {"submission_file": ("submission.txt", b"This includes evidence and analysis.", "text/plain")}
    data = {"assignment_instructions": "Write a written assignment with argument and evidence."}

    response = client.post("/grade", files=files, data=data)
    assert response.status_code == 200

    payload = response.json()
    assert payload["assignment_type"] == "written assignment"
    assert payload["total_score"] <= payload["max_score"]
    assert len(payload["criteria_results"]) == 5
    assert all("level" in result for result in payload["criteria_results"])

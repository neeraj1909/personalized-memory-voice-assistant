from pathlib import Path

from scripts import evaluate_golden as evaluator


def test_context_checks_report_missing_and_excluded_keys():
    turn = {"expected_context_keys": ["preference"], "excluded_context_keys": ["other_user"]}
    assert evaluator.assess_context(turn, {"context_used": ["other_user"]}) == {
        "pass": False,
        "missing": ["preference"],
        "excluded_found": ["other_user"],
    }


def test_modes_use_temporary_databases_and_preserve_turn_order(monkeypatch):
    paths = []
    messages = []

    class Response:
        status_code = 200

        def json(self):
            return {"context_used": [], "response": "A response"}

    class Client:
        def __init__(self, app):
            paths.append(app.database_path)

        def __enter__(self):
            assert paths[-1].parent.exists()
            return self

        def __exit__(self, *_):
            pass

        def post(self, path, json):
            assert path == "/chat"
            messages.append(json["message"])
            return Response()

    monkeypatch.setattr(evaluator, "create_app", lambda *, settings: settings)
    monkeypatch.setattr(evaluator, "TestClient", Client)
    turn = {"user_id": "synthetic", "session_id": "one"}
    scenarios = [
        {"name": "sample", "turns": [{**turn, "message": "first"}, {**turn, "message": "second"}]}
    ]
    monkeypatch.setenv("PMVA_MEMORY_FAST", "existing")
    for mode in ("original", "fast"):
        result = evaluator.run_mode(mode, scenarios, {}, False)
        assert result["successful_turns"] == 2
        assert result["context_checks_total"] == 0
    assert messages == ["first", "second", "first", "second"]
    assert paths[0] != paths[1]
    assert all(not Path(path).exists() for path in paths)
    assert evaluator.os.environ["PMVA_MEMORY_FAST"] == "existing"

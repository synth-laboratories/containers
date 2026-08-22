from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from synth_containers.alfworld import app as alfworld


@dataclass
class _State:
    observation: str = "A room"
    terminal: bool = False
    won: bool = False


class _Runtime:
    @staticmethod
    def goal_progress() -> float:
        return 0.0


class _Game:
    runtime = _Runtime()

    def __init__(self) -> None:
        self.state = _State()
        self.updates: list[str] = []

    def reset(self) -> _State:
        return self.state

    @staticmethod
    def available_actions() -> list[str]:
        return ["look", "go north"]

    def update(self, action: str) -> _State:
        self.updates.append(action)
        return self.state


class _Response:
    is_error = False

    @staticmethod
    def json() -> dict:
        return {
            "text": "",
            "prompt_token_ids": [1, 2],
            "token_ids": [],
            "log_probs": [],
        }


class _AsyncClient:
    def __init__(self, **_kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def post(self, *_args, **_kwargs) -> _Response:
        return _Response()


def test_empty_sample_is_recorded_as_failed_generation(monkeypatch, tmp_path) -> None:
    game = _Game()
    monkeypatch.setattr(alfworld, "_rows", lambda split=None: [{"id": "row-1"}])
    monkeypatch.setattr(alfworld, "_game", lambda _row: game)
    monkeypatch.setattr(alfworld.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(alfworld, "RUN_ROOT", tmp_path)

    response = TestClient(alfworld.app).post(
        "/training/rollouts",
        json={
            "schema_version": alfworld.TRAINING_REQUEST_VERSION,
            "rollout_id": "empty-generation",
            "policy_version": "snap_test",
            "task": {"task_id": "alfworld.text.v1", "max_turns": 2},
            "sampler": {"url": "http://sampler.invalid/sample"},
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["trajectory"] == [alfworld.INVALID_EMPTY_ACTION]
    assert result["reward"] == {"reward": 0.0, "won": False}
    assert result["reward_info"]["invalid_completion"] is True
    assert result["reward_info"]["exploration_coverage"] == 0.0
    assert game.updates == []

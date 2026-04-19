from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ArtifactRef, FeedbackEvent, GoalSpec, PlanVersion, RunState


class FilePlannerStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_goal(self, run_id: str, goal: GoalSpec) -> Path:
        path = self._run_dir(run_id) / "goal.json"
        self._write_json(path, goal.to_dict())
        return path

    def load_goal(self, run_id: str) -> GoalSpec:
        return GoalSpec.from_dict(self._read_json(self._run_dir(run_id) / "goal.json"))

    def save_state(self, run_id: str, state: RunState) -> Path:
        path = self._run_dir(run_id) / "state.json"
        self._write_json(path, state.to_dict())
        return path

    def load_state(self, run_id: str) -> RunState:
        return RunState.from_dict(self._read_json(self._run_dir(run_id) / "state.json"))

    def save_plan(self, run_id: str, plan: PlanVersion) -> Path:
        path = self._run_dir(run_id) / "plans" / f"v{plan.version}.json"
        self._write_json(path, plan.to_dict())
        return path

    def load_plan(self, run_id: str, version: int) -> PlanVersion:
        return PlanVersion.from_dict(self._read_json(self._run_dir(run_id) / "plans" / f"v{version}.json"))

    def list_plan_versions(self, run_id: str) -> list[int]:
        plans_dir = self._run_dir(run_id) / "plans"
        if not plans_dir.exists():
            return []
        versions = [int(path.stem[1:]) for path in plans_dir.glob("v*.json")]
        return sorted(versions)

    def save_artifact(self, run_id: str, artifact: ArtifactRef, payload: Any) -> Path:
        path = self._run_dir(run_id) / "artifacts" / f"{artifact.artifact_id}.json"
        self._write_json(path, {"artifact": artifact.to_dict(), "payload": payload})
        return path

    def load_artifact(self, run_id: str, artifact_id: str) -> tuple[ArtifactRef, Any]:
        payload = self._read_json(self._run_dir(run_id) / "artifacts" / f"{artifact_id}.json")
        return ArtifactRef.from_dict(payload["artifact"]), payload["payload"]

    def load_artifacts(self, run_id: str, artifact_ids: list[str]) -> list[ArtifactRef]:
        return [self.load_artifact(run_id, artifact_id)[0] for artifact_id in artifact_ids]

    def append_feedback(self, run_id: str, event: FeedbackEvent) -> Path:
        path = self._run_dir(run_id) / "feedback" / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True))
            handle.write("\n")
        return path

    def load_feedback(self, run_id: str) -> list[FeedbackEvent]:
        path = self._run_dir(run_id) / "feedback" / "events.jsonl"
        if not path.exists():
            return []
        events: list[FeedbackEvent] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(FeedbackEvent.from_dict(json.loads(line)))
        return events

    def _run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _read_json(self, path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

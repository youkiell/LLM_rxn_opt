import json
from pathlib import Path
from typing import Any, Dict, List


class MemoryStore:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.agent_path = self.output_dir / "agent_memory.jsonl"
        self.coordinator_path = self.output_dir / "coordinator_memory.jsonl"
        self.summary_path = self.output_dir / "meeting_summary.jsonl"
        self.experiment_path = self.output_dir / "experiment_memory.jsonl"

    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_agent_message(
        self,
        round_id: int,
        reaction_key: tuple,
        agent_name: str,
        payload: Dict[str, Any],
    ) -> None:
        record = {
            "round_id": round_id,
            "reaction_key": list(reaction_key),
            "agent_name": agent_name,
            "payload": payload,
        }
        self._append_jsonl(self.agent_path, record)

    def log_coordinator_decision(
        self,
        round_id: int,
        reaction_key: tuple,
        payload: Dict[str, Any],
    ) -> None:
        record = {
            "round_id": round_id,
            "reaction_key": list(reaction_key),
            "payload": payload,
        }
        self._append_jsonl(self.coordinator_path, record)

    def log_meeting_summary(
        self,
        round_id: int,
        reaction_key: tuple,
        payload: Dict[str, Any],
    ) -> None:
        record = {
            "round_id": round_id,
            "reaction_key": list(reaction_key),
            "payload": payload,
        }
        self._append_jsonl(self.summary_path, record)

    def log_experiment_result(
        self,
        round_id: int,
        reaction_key: tuple,
        selected_action: str,
        observed_target: float,
        stop_hit: bool,
    ) -> None:
        record = {
            "round_id": round_id,
            "reaction_key": list(reaction_key),
            "selected_action": selected_action,
            "observed_target": observed_target,
            "stop_hit": stop_hit,
        }
        self._append_jsonl(self.experiment_path, record)

    def load_experiment_rows(self) -> List[Dict[str, Any]]:
        if not self.experiment_path.exists():
            return []

        rows = []
        with self.experiment_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["reaction_key"] = tuple(row["reaction_key"])
                    rows.append(row)
        return rows
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkConfig:
    csv_path: str
    output_dir: str
    n_reactions: int = 10
    num_rounds: int = 8
    stop_threshold: float = 90.0
    model_name: str = "gpt-5.4"
    active_agents: List[str] = field(default_factory=list)
    key_columns: List[str] = field(default_factory=list)
    action_column: str = "action"
    target_column: str = "target"
    cluster_column: Optional[str] = None
    top_k_per_agent: int = 5
    temperature: float = 0.2
    random_seed: int = 42


@dataclass
class ReactionTask:
    task_id: str
    reaction_key: tuple
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeetingMessage:
    round_id: int
    agent_name: str
    content: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeetingState:
    config: BenchmarkConfig
    task: ReactionTask
    env: Dict[str, Any]
    memory_rows: List[Dict[str, Any]]
    selected_this_round: List[str] = field(default_factory=list)
    transcript: List[MeetingMessage] = field(default_factory=list)
    final_choice: Optional[Dict[str, Any]] = None
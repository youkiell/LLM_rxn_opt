from pathlib import Path

import numpy as np
import pandas as pd

from state import MeetingMessage, MeetingState, ReactionTask
from orchestration.agent_registry import get_active_agents
from utils.data_utils import choose_reactions, load_environment
from memory.store import MemoryStore


def _safe_top_action(agent_payload):
    proposals = agent_payload.get("proposals", [])
    if not proposals:
        return None
    return proposals[0].get("candidate")


def run_experiment(config):
    env = load_environment(config)
    reactions = choose_reactions(env, config)
    active = list(reactions)

    memory_store = MemoryStore(config.output_dir)
    memory_rows = []

    detailed_logs = []
    agents = get_active_agents(config)

    for round_id in range(1, config.num_rounds + 1):
        if not active:
            break

        selected_this_round = []
        batch_rows = []

        for reaction_key in active:
            state = MeetingState(
                config=config,
                task=ReactionTask(
                    task_id=f"round{round_id}_{str(reaction_key)}",
                    reaction_key=reaction_key,
                ),
                env=env,
                memory_rows=memory_rows,
                selected_this_round=selected_this_round,
            )

            for agent_name in config.active_agents:
                if agent_name in {"coordinator_agent", "summary_agent"}:
                    continue

                result = agents[agent_name].respond(state)
                state.transcript.append(
                    MeetingMessage(
                        round_id=round_id,
                        agent_name=agent_name,
                        content=result.get("content", ""),
                        payload=result.get("payload", {}),
                    )
                )

                memory_store.log_agent_message(
                    round_id=round_id,
                    reaction_key=reaction_key,
                    agent_name=agent_name,
                    payload=result.get("payload", {}),
                )

            coord_result = agents["coordinator_agent"].respond(state)
            state.transcript.append(
                MeetingMessage(
                    round_id=round_id,
                    agent_name="coordinator_agent",
                    content=coord_result.get("content", ""),
                    payload=coord_result.get("payload", {}),
                )
            )

            memory_store.log_coordinator_decision(
                round_id=round_id,
                reaction_key=reaction_key,
                payload=coord_result.get("payload", {}),
            )

            if "summary_agent" in agents:
                summary_result = agents["summary_agent"].respond(state)
                state.transcript.append(
                    MeetingMessage(
                        round_id=round_id,
                        agent_name="summary_agent",
                        content=summary_result.get("content", ""),
                        payload=summary_result.get("payload", {}),
                    )
                )

                memory_store.log_meeting_summary(
                    round_id=round_id,
                    reaction_key=reaction_key,
                    payload=summary_result.get("payload", {}),
                )
            else:
                summary_result = {"payload": {}}

            chosen_action = coord_result["payload"]["selected_action"]
            selected_this_round.append(chosen_action)

            truth_key = reaction_key + (chosen_action,)
            observed = env["truth"][truth_key]
            stop_hit = observed >= config.stop_threshold

            row_data = {
                "round_id": round_id,
                "reaction_key": reaction_key,
                config.action_column: chosen_action,
                config.target_column: observed,
                "stop_hit": stop_hit,
            }

            for i, col in enumerate(config.key_columns):
                row_data[col] = reaction_key[i]

            batch_rows.append(row_data)

            memory_store.log_experiment_result(
                round_id=round_id,
                reaction_key=reaction_key,
                selected_action=chosen_action,
                observed_target=observed,
                stop_hit=stop_hit,
            )

            detailed_logs.append({
                "round_id": round_id,
                "reaction_key": str(reaction_key),
                "chosen_action": chosen_action,
                "observed_target": observed,
                "stop_hit": stop_hit,
                "existing_top": _safe_top_action(
                    next((m.payload for m in state.transcript if m.agent_name == "existing_reaction_agent"), {})
                ),
                "exploring_top": _safe_top_action(
                    next((m.payload for m in state.transcript if m.agent_name == "exploring_reaction_agent"), {})
                ),
                "coordinator_reason": coord_result["payload"].get("rationale", ""),
                "meeting_summary": summary_result["payload"].get("summary", ""),
            })

        if batch_rows:
            memory_rows.extend(batch_rows)

        solved = set()
        for row in batch_rows:
            if row["stop_hit"]:
                solved.add(row["reaction_key"])

        active = [rk for rk in active if rk not in solved]

    if not detailed_logs:
        raise RuntimeError("No logs were created.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_df = pd.DataFrame(detailed_logs)
    memory_df = pd.DataFrame(memory_rows)

    summary_df = (
        log_df.groupby(["reaction_key"], as_index=False)
        .agg(
            best_target=("observed_target", "max"),
            n_trials=("observed_target", "count"),
        )
    )
    summary_df["hit_threshold"] = summary_df["best_target"] >= config.stop_threshold

    first_success = (
        log_df[log_df["stop_hit"]]
        .sort_values("round_id")
        .groupby(["reaction_key"], as_index=False)
        .first()[["reaction_key", "round_id"]]
        .rename(columns={"round_id": "round_of_first_success"})
    )

    summary_df = summary_df.merge(first_success, on="reaction_key", how="left")

    overall_df = pd.DataFrame([{
        "n_reactions": len(reactions),
        "n_solved": int(summary_df["hit_threshold"].sum()),
        "solve_rate": float(summary_df["hit_threshold"].mean()),
        "mean_best_target": float(summary_df["best_target"].mean()),
        "mean_trials": float(summary_df["n_trials"].mean()),
        "mean_round_of_first_success": float(summary_df["round_of_first_success"].dropna().mean())
        if summary_df["round_of_first_success"].notna().any() else np.nan,
    }])

    log_df.to_csv(output_dir / "llm_multi_agent_log.csv", index=False)
    summary_df.to_csv(output_dir / "llm_multi_agent_per_reaction_summary.csv", index=False)
    overall_df.to_csv(output_dir / "llm_multi_agent_overall_summary.csv", index=False)
    memory_df.to_csv(output_dir / "llm_memory_trace.csv", index=False)

    print("Finished LLM multi-agent run.")
    print(f"Results written to: {output_dir}")
    print(overall_df.to_string(index=False))
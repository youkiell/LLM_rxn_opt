from pathlib import Path
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def make_reaction_key(row: pd.Series, key_columns: List[str]) -> tuple:
    return tuple(row[c] for c in key_columns)


def load_environment(config) -> Dict:
    df = pd.read_csv(config.csv_path).copy()

    required = list(config.key_columns) + [config.action_column, config.target_column]
    if config.cluster_column:
        required.append(config.cluster_column)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["reaction_key"] = df.apply(
        lambda row: make_reaction_key(row, config.key_columns), axis=1
    )

    truth = {}
    for _, row in df.iterrows():
        truth[row["reaction_key"] + (row[config.action_column],)] = float(row[config.target_column])

    reaction_to_candidates = (
        df.groupby("reaction_key")[config.action_column]
        .apply(lambda x: sorted(x.astype(str).unique().tolist()))
        .to_dict()
    )

    reaction_to_cluster = {}
    if config.cluster_column:
        reaction_to_cluster = (
            df.groupby("reaction_key")[config.cluster_column]
            .first()
            .to_dict()
        )

    global_action_mean = (
        df.groupby(config.action_column)[config.target_column]
        .mean()
        .to_dict()
    )

    return {
        "df": df,
        "truth": truth,
        "reaction_to_candidates": reaction_to_candidates,
        "reaction_to_cluster": reaction_to_cluster,
        "global_action_mean": global_action_mean,
    }


def choose_reactions(env: Dict, config) -> List[tuple]:
    reaction_df = env["df"][["reaction_key"]].drop_duplicates().copy()

    if config.cluster_column:
        cluster_map = env["df"][["reaction_key", config.cluster_column]].drop_duplicates()
        reaction_df = reaction_df.merge(cluster_map, on="reaction_key", how="left")
        reaction_df = reaction_df.sort_values([config.cluster_column, "reaction_key"])

        chosen = []
        seen_clusters = set()

        for _, row in reaction_df.iterrows():
            rk = row["reaction_key"]
            cl = row[config.cluster_column]
            if cl not in seen_clusters:
                chosen.append(rk)
                seen_clusters.add(cl)
            if len(chosen) >= config.n_reactions:
                return chosen

        for _, row in reaction_df.iterrows():
            rk = row["reaction_key"]
            if rk not in chosen:
                chosen.append(rk)
            if len(chosen) >= config.n_reactions:
                return chosen

        return chosen

    keys = reaction_df["reaction_key"].tolist()
    rng = random.Random(config.random_seed)
    rng.shuffle(keys)
    return keys[: config.n_reactions]


def get_tested_for_reaction(memory_df: pd.DataFrame, reaction_key: tuple, action_column: str) -> set:
    if memory_df.empty:
        return set()
    mask = memory_df["reaction_key"] == reaction_key
    return set(memory_df.loc[mask, action_column].astype(str).tolist())


def available_actions(env: Dict, memory_df: pd.DataFrame, reaction_key: tuple, action_column: str) -> List[str]:
    used = get_tested_for_reaction(memory_df, reaction_key, action_column)
    return [a for a in env["reaction_to_candidates"][reaction_key] if a not in used]


def row_overlap_score(
    row_key: tuple,
    target_key: tuple,
) -> int:
    return sum(int(a == b) for a, b in zip(row_key, target_key))


def build_memory_dataframe(memory_rows: List[Dict]) -> pd.DataFrame:
    if not memory_rows:
        return pd.DataFrame()
    return pd.DataFrame(memory_rows)


def summarize_candidate_features(
    env: Dict,
    memory_df: pd.DataFrame,
    reaction_key: tuple,
    candidate: str,
    config,
) -> Dict:
    pred = float(env["global_action_mean"].get(candidate, 0.0) / 100.0)

    if memory_df.empty:
        return {
            "candidate": candidate,
            "predicted_success": pred,
            "uncertainty": 1.0,
            "transferability": 0.0,
            "related_observations": [],
        }

    related_rows = []
    for _, row in memory_df.iterrows():
        mem_key = row["reaction_key"]
        overlap = row_overlap_score(mem_key, reaction_key)
        if row[config.action_column] == candidate and overlap > 0:
            related_rows.append(
                {
                    "reaction_key": mem_key,
                    "observed_target": float(row[config.target_column]),
                    "overlap": overlap,
                }
            )

    if related_rows:
        best_overlap = max(r["overlap"] for r in related_rows)
        same_band = [r for r in related_rows if r["overlap"] == best_overlap]
        transfer = float(np.mean([r["observed_target"] for r in same_band]) / 100.0)
        uncertainty = float(np.exp(-0.7 * len(related_rows)))
    else:
        transfer = 0.0
        uncertainty = 1.0

    return {
        "candidate": candidate,
        "predicted_success": pred,
        "uncertainty": uncertainty,
        "transferability": transfer,
        "related_observations": related_rows[:10],
    }
#!/usr/bin/env python3
"""
starter_multi_agent_catalyst_v2.py

A simple, tested starting point for a synchronous multi-agent catalyst search
simulation using Filtered_Virtual_Predictions.csv.

This version is intentionally pragmatic:
- no LLM calls
- no personas
- role behavior is implemented as different scoring functions
- all selections in a round are made from prior-round memory only
- outcomes are revealed together at the end of the round
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROUNDS = 8
DEFAULT_STOP_THRESHOLD = 90.0
DEFAULT_TOP_K = 5
DEFAULT_WEIGHTS = {
    "alpha": 0.45,  # predicted success
    "beta": 0.20,   # uncertainty
    "gamma": 0.20,  # transferability
    "delta": 0.10,  # diversity gain
    "lam": 0.25,    # redundancy penalty
}


def load_environment(csv_path):
    df = pd.read_csv(csv_path).copy()

    required = ["Imine", "Nucleophile", "Catalyst_Ar_grp", "cluster_label", "Predicted ee"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["reaction_key"] = list(zip(df["Imine"], df["Nucleophile"]))

    truth = {}
    for _, row in df.iterrows():
        truth[(row["Imine"], row["Nucleophile"], row["Catalyst_Ar_grp"])] = float(row["Predicted ee"])

    reaction_to_candidates = (
        df.groupby("reaction_key")["Catalyst_Ar_grp"]
        .apply(lambda x: sorted(x.unique().tolist()))
        .to_dict()
    )

    reaction_to_cluster = (
        df.groupby("reaction_key")["cluster_label"]
        .first()
        .to_dict()
    )

    global_catalyst_mean = (
        df.groupby("Catalyst_Ar_grp")["Predicted ee"]
        .mean()
        .to_dict()
    )

    return {
        "df": df,
        "truth": truth,
        "reaction_to_candidates": reaction_to_candidates,
        "reaction_to_cluster": reaction_to_cluster,
        "global_catalyst_mean": global_catalyst_mean,
    }


def choose_reactions(env, n_reactions):
    reaction_df = (
        env["df"][["Imine", "Nucleophile", "cluster_label"]]
        .drop_duplicates()
        .sort_values(["cluster_label", "Imine", "Nucleophile"])
        .reset_index(drop=True)
    )

    chosen = []
    seen_clusters = set()

    # First pass: spread across clusters
    for _, row in reaction_df.iterrows():
        key = (row["Imine"], row["Nucleophile"])
        cl = int(row["cluster_label"])
        if cl not in seen_clusters:
            chosen.append(key)
            seen_clusters.add(cl)
        if len(chosen) >= n_reactions:
            return chosen

    # Second pass: fill remaining
    for _, row in reaction_df.iterrows():
        key = (row["Imine"], row["Nucleophile"])
        if key not in chosen:
            chosen.append(key)
        if len(chosen) >= n_reactions:
            return chosen

    return chosen


def get_tested_for_reaction(memory_df, reaction_key):
    if memory_df.empty:
        return set()
    mask = (
        (memory_df["Imine"] == reaction_key[0]) &
        (memory_df["Nucleophile"] == reaction_key[1])
    )
    return set(memory_df.loc[mask, "Catalyst_Ar_grp"].tolist())


def available_catalysts(env, memory_df, reaction_key):
    used = get_tested_for_reaction(memory_df, reaction_key)
    return [c for c in env["reaction_to_candidates"][reaction_key] if c not in used]


def predicted_success(env, memory_df, reaction_key, catalyst):
    # 1) same reaction prior evidence
    if not memory_df.empty:
        mask = (
            (memory_df["Imine"] == reaction_key[0]) &
            (memory_df["Nucleophile"] == reaction_key[1]) &
            (memory_df["Catalyst_Ar_grp"] == catalyst)
        )
        same_rxn = memory_df.loc[mask, "Predicted ee"]
        if len(same_rxn) > 0:
            return float(same_rxn.mean() / 100.0), "same reaction evidence"

    # 2) same cluster
    if not memory_df.empty:
        cluster_id = env["reaction_to_cluster"][reaction_key]
        tmp = memory_df.copy()
        tmp["reaction_key"] = list(zip(tmp["Imine"], tmp["Nucleophile"]))
        tmp["cluster_label"] = tmp["reaction_key"].map(env["reaction_to_cluster"])
        mask = (
            (tmp["cluster_label"] == cluster_id) &
            (tmp["Catalyst_Ar_grp"] == catalyst)
        )
        same_cluster = tmp.loc[mask, "Predicted ee"]
        if len(same_cluster) > 0:
            return float(same_cluster.mean() / 100.0), "same cluster evidence"

    # 3) global prior
    return float(env["global_catalyst_mean"][catalyst] / 100.0), "global prior"


def uncertainty(env, memory_df, reaction_key, catalyst):
    if memory_df.empty:
        return 1.0, "no observations yet"

    same_rxn_mask = (
        (memory_df["Imine"] == reaction_key[0]) &
        (memory_df["Nucleophile"] == reaction_key[1]) &
        (memory_df["Catalyst_Ar_grp"] == catalyst)
    )
    if same_rxn_mask.sum() > 0:
        return 0.1, "already tested in this reaction"

    tmp = memory_df.copy()
    tmp["reaction_key"] = list(zip(tmp["Imine"], tmp["Nucleophile"]))
    tmp["cluster_label"] = tmp["reaction_key"].map(env["reaction_to_cluster"])
    cluster_id = env["reaction_to_cluster"][reaction_key]
    n_cluster = int(((tmp["cluster_label"] == cluster_id) & (tmp["Catalyst_Ar_grp"] == catalyst)).sum())

    score = float(np.exp(-0.7 * n_cluster))
    return score, f"{n_cluster} same-cluster observations"


def transferability(env, memory_df, reaction_key, catalyst):
    if memory_df.empty:
        return 0.0, "no transfer evidence"

    tmp = memory_df.copy()
    tmp["reaction_key"] = list(zip(tmp["Imine"], tmp["Nucleophile"]))
    tmp["cluster_label"] = tmp["reaction_key"].map(env["reaction_to_cluster"])
    cluster_id = env["reaction_to_cluster"][reaction_key]

    same_cluster = tmp[
        (tmp["cluster_label"] == cluster_id) &
        (tmp["Catalyst_Ar_grp"] == catalyst)
    ]
    if len(same_cluster) > 0:
        return float(same_cluster["Predicted ee"].mean() / 100.0), "same cluster"

    same_imine_or_nuc = memory_df[
        (
            (memory_df["Imine"] == reaction_key[0]) |
            (memory_df["Nucleophile"] == reaction_key[1])
        ) &
        (memory_df["Catalyst_Ar_grp"] == catalyst)
    ]
    if len(same_imine_or_nuc) > 0:
        return float(same_imine_or_nuc["Predicted ee"].mean() / 100.0), "same imine or nucleophile"

    return 0.0, "no related precedent"


def diversity_gain(catalyst, selected_this_round):
    # Placeholder until catalyst descriptors are added.
    if catalyst in selected_this_round:
        return 0.0, "duplicate catalyst in batch"
    return 1.0, "new catalyst in batch"


def redundancy(memory_df, reaction_key, catalyst):
    used = get_tested_for_reaction(memory_df, reaction_key)
    if catalyst in used:
        return 1.0, "already tested for this reaction"
    return 0.0, "not yet tested for this reaction"


def build_candidate_row(env, memory_df, reaction_key, catalyst, selected_this_round):
    pred, pred_note = predicted_success(env, memory_df, reaction_key, catalyst)
    unc, unc_note = uncertainty(env, memory_df, reaction_key, catalyst)
    trans, trans_note = transferability(env, memory_df, reaction_key, catalyst)
    div, div_note = diversity_gain(catalyst, selected_this_round)
    red, red_note = redundancy(memory_df, reaction_key, catalyst)

    return {
        "catalyst": catalyst,
        "predicted_success": pred,
        "uncertainty": unc,
        "transferability": trans,
        "diversity_gain": div,
        "redundancy": red,
        "pred_note": pred_note,
        "unc_note": unc_note,
        "trans_note": trans_note,
        "div_note": div_note,
        "red_note": red_note,
    }


def propose_explorer(env, memory_df, reaction_key, selected_this_round):
    rows = []
    for c in available_catalysts(env, memory_df, reaction_key):
        row = build_candidate_row(env, memory_df, reaction_key, c, selected_this_round)
        row["agent_score"] = (
            0.20 * row["predicted_success"]
            + 0.55 * row["uncertainty"]
            + 0.10 * row["transferability"]
            + 0.10 * row["diversity_gain"]
            - 0.05 * row["redundancy"]
        )
        row["role"] = "explorer"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("agent_score", ascending=False)


def propose_exploiter(env, memory_df, reaction_key, selected_this_round):
    rows = []
    for c in available_catalysts(env, memory_df, reaction_key):
        row = build_candidate_row(env, memory_df, reaction_key, c, selected_this_round)
        row["agent_score"] = (
            0.65 * row["predicted_success"]
            + 0.05 * row["uncertainty"]
            + 0.20 * row["transferability"]
            + 0.05 * row["diversity_gain"]
            - 0.05 * row["redundancy"]
        )
        row["role"] = "exploiter"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("agent_score", ascending=False)


def propose_transferability(env, memory_df, reaction_key, selected_this_round):
    rows = []
    for c in available_catalysts(env, memory_df, reaction_key):
        row = build_candidate_row(env, memory_df, reaction_key, c, selected_this_round)
        row["agent_score"] = (
            0.20 * row["predicted_success"]
            + 0.05 * row["uncertainty"]
            + 0.65 * row["transferability"]
            + 0.05 * row["diversity_gain"]
            - 0.05 * row["redundancy"]
        )
        row["role"] = "transferability"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("agent_score", ascending=False)


def propose_critic(env, memory_df, reaction_key, selected_this_round):
    rows = []
    for c in available_catalysts(env, memory_df, reaction_key):
        row = build_candidate_row(env, memory_df, reaction_key, c, selected_this_round)
        row["agent_score"] = (
            0.25 * row["predicted_success"]
            + 0.25 * row["uncertainty"]
            + 0.20 * row["transferability"]
            + 0.20 * row["diversity_gain"]
            - 0.10 * row["redundancy"]
        )
        row["role"] = "critic"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("agent_score", ascending=False)


def coordinator_select(proposals, selected_this_round, weights):
    pooled = pd.concat(proposals, ignore_index=True)
    grouped = (
        pooled.groupby("catalyst", as_index=False)[
            ["predicted_success", "uncertainty", "transferability", "redundancy"]
        ]
        .mean()
    )

    div_scores = []
    div_notes = []
    for c in grouped["catalyst"]:
        div, note = diversity_gain(c, selected_this_round)
        div_scores.append(div)
        div_notes.append(note)

    grouped["diversity_gain"] = div_scores
    grouped["div_note"] = div_notes
    grouped["coord_score"] = (
        weights["alpha"] * grouped["predicted_success"]
        + weights["beta"] * grouped["uncertainty"]
        + weights["gamma"] * grouped["transferability"]
        + weights["delta"] * grouped["diversity_gain"]
        - weights["lam"] * grouped["redundancy"]
    )

    grouped = grouped.sort_values("coord_score", ascending=False).reset_index(drop=True)
    return grouped.iloc[0].to_dict()


def run_experiment(csv_path, output_dir, n_reactions, num_rounds, stop_threshold):
    env = load_environment(csv_path)
    reactions = choose_reactions(env, n_reactions)
    active = list(reactions)

    memory_df = pd.DataFrame(columns=[
        "round_id", "Imine", "Nucleophile", "Catalyst_Ar_grp", "Predicted ee", "stop_hit"
    ])
    detailed_logs = []

    for round_id in range(1, num_rounds + 1):
        if not active:
            break

        selected_this_round = []
        batch_rows = []

        for reaction_key in active:
            avail = available_catalysts(env, memory_df, reaction_key)
            if not avail:
                continue

            explorer_df = propose_explorer(env, memory_df, reaction_key, selected_this_round)
            exploiter_df = propose_exploiter(env, memory_df, reaction_key, selected_this_round)
            transfer_df = propose_transferability(env, memory_df, reaction_key, selected_this_round)
            critic_df = propose_critic(env, memory_df, reaction_key, selected_this_round)

            chosen = coordinator_select(
                [explorer_df.head(DEFAULT_TOP_K),
                 exploiter_df.head(DEFAULT_TOP_K),
                 transfer_df.head(DEFAULT_TOP_K),
                 critic_df.head(DEFAULT_TOP_K)],
                selected_this_round,
                DEFAULT_WEIGHTS,
            )

            catalyst = chosen["catalyst"]
            selected_this_round.append(catalyst)

            ee = env["truth"][(reaction_key[0], reaction_key[1], catalyst)]
            stop_hit = ee >= stop_threshold

            batch_rows.append({
                "round_id": round_id,
                "Imine": reaction_key[0],
                "Nucleophile": reaction_key[1],
                "Catalyst_Ar_grp": catalyst,
                "Predicted ee": ee,
                "stop_hit": stop_hit,
            })

            detailed_logs.append({
                "round_id": round_id,
                "Imine": reaction_key[0],
                "Nucleophile": reaction_key[1],
                "cluster_label": env["reaction_to_cluster"][reaction_key],
                "chosen_catalyst": catalyst,
                "observed_ee": ee,
                "stop_hit": stop_hit,
                "coord_score": chosen["coord_score"],
                "coord_predicted_success": chosen["predicted_success"],
                "coord_uncertainty": chosen["uncertainty"],
                "coord_transferability": chosen["transferability"],
                "coord_diversity_gain": chosen["diversity_gain"],
                "coord_redundancy": chosen["redundancy"],
                "explorer_top": explorer_df.iloc[0]["catalyst"] if len(explorer_df) else None,
                "exploiter_top": exploiter_df.iloc[0]["catalyst"] if len(exploiter_df) else None,
                "transferability_top": transfer_df.iloc[0]["catalyst"] if len(transfer_df) else None,
                "critic_top": critic_df.iloc[0]["catalyst"] if len(critic_df) else None,
            })

        if batch_rows:
            # Synchronous reveal: add entire round only after all choices are made.
            memory_df = pd.concat([memory_df, pd.DataFrame(batch_rows)], ignore_index=True)

        solved = set()
        for row in batch_rows:
            if row["stop_hit"]:
                solved.add((row["Imine"], row["Nucleophile"]))

        active = [rk for rk in active if rk not in solved]

    if not detailed_logs:
        raise RuntimeError("No logs were created.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_df = pd.DataFrame(detailed_logs)

    first_success = (
        log_df[log_df["stop_hit"]]
        .sort_values("round_id")
        .groupby(["Imine", "Nucleophile"], as_index=False)
        .first()[["Imine", "Nucleophile", "round_id"]]
        .rename(columns={"round_id": "round_of_first_success"})
    )

    summary_df = (
        log_df.groupby(["Imine", "Nucleophile"], as_index=False)
        .agg(best_ee=("observed_ee", "max"),
             n_trials=("observed_ee", "count"))
    )
    summary_df["hit_threshold"] = summary_df["best_ee"] >= stop_threshold
    summary_df = summary_df.merge(first_success, on=["Imine", "Nucleophile"], how="left")

    overall_df = pd.DataFrame([{
        "n_reactions": len(reactions),
        "n_solved": int(summary_df["hit_threshold"].sum()),
        "solve_rate": float(summary_df["hit_threshold"].mean()),
        "mean_best_ee": float(summary_df["best_ee"].mean()),
        "mean_trials": float(summary_df["n_trials"].mean()),
        "mean_round_of_first_success": float(summary_df["round_of_first_success"].dropna().mean()) if summary_df["round_of_first_success"].notna().any() else np.nan,
    }])

    log_df.to_csv(output_dir / "starter_multi_agent_log.csv", index=False)
    summary_df.to_csv(output_dir / "starter_multi_agent_per_reaction_summary.csv", index=False)
    overall_df.to_csv(output_dir / "starter_multi_agent_overall_summary.csv", index=False)

    print("Finished starter synchronous multi-agent run.")
    print(f"Results written to: {output_dir}")
    print(overall_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="Filtered_Virtual_Predictions.csv")
    parser.add_argument("--output_dir", type=str, default="starter_results")
    parser.add_argument("--n_reactions", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--stop_threshold", type=float, default=DEFAULT_STOP_THRESHOLD)
    args = parser.parse_args()

    run_experiment(
        csv_path=args.csv,
        output_dir=args.output_dir,
        n_reactions=args.n_reactions,
        num_rounds=args.num_rounds,
        stop_threshold=args.stop_threshold,
    )


if __name__ == "__main__":
    main()

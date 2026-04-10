from state import BenchmarkConfig
from orchestration.meeting_runner import run_experiment


ACTIVE_AGENTS = [
    "coordinator_agent",
    "existing_reaction_agent",
    "exploring_reaction_agent",
    "summary_agent",
    # "transferability_agent",
    # "critic_agent",
]


def main():
    config = BenchmarkConfig(
        csv_path="Filtered_Virtual_Predictions.csv",
        output_dir="llm_results",
        n_reactions=10,
        num_rounds=8,
        stop_threshold=90.0,
        model_name="gpt-5.4",
        active_agents=ACTIVE_AGENTS,
        key_columns=["Imine", "Nucleophile"],
        action_column="Catalyst_Ar_grp",
        target_column="Predicted ee",
        cluster_column="cluster_label",
        top_k_per_agent=5,
        temperature=0.2,
        random_seed=42,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
from agents.base_agent import BaseAgent
from utils.io import load_prompt
from utils.llm_client import LLMClient
from utils.data_utils import build_memory_dataframe, available_actions, summarize_candidate_features


class ExploringReactionAgent(BaseAgent):
    name = "exploring_reaction_agent"

    def __init__(self, config):
        self.prompt = load_prompt("exploring_reaction_agent")
        self.llm = LLMClient(
            model_name=config.model_name,
            temperature=config.temperature,
        )
        self.config = config

    def respond(self, state):
        memory_df = build_memory_dataframe(state.memory_rows)

        candidates = available_actions(
            state.env,
            memory_df,
            state.task.reaction_key,
            self.config.action_column,
        )

        candidate_summaries = []
        for c in candidates:
            summary = summarize_candidate_features(
                state.env,
                memory_df,
                state.task.reaction_key,
                c,
                self.config,
            )
            candidate_summaries.append(summary)

        payload = {
            "agent_role": "exploring_reaction_agent",
            "reaction_key": list(state.task.reaction_key),
            "candidate_summaries": candidate_summaries[:40],
            "top_k": self.config.top_k_per_agent,
            "instruction": "Allow broader, higher-uncertainty suggestions if they look promising."
        }

        result = self.llm.generate_json(self.prompt, payload)
        return {
            "content": result.get("rationale", ""),
            "payload": result,
        }
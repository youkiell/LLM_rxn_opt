from agents.base_agent import BaseAgent
from utils.io import load_prompt
from utils.llm_client import LLMClient


class CoordinatorAgent(BaseAgent):
    name = "coordinator_agent"

    def __init__(self, config):
        self.prompt = load_prompt("coordinator")
        self.llm = LLMClient(
            model_name=config.model_name,
            temperature=config.temperature,
        )
        self.config = config

    def respond(self, state):
        agent_outputs = {}
        for msg in state.transcript:
            if msg.agent_name in {"existing_reaction_agent", "exploring_reaction_agent"}:
                agent_outputs[msg.agent_name] = msg.payload

        payload = {
            "reaction_key": list(state.task.reaction_key),
            "top_k": self.config.top_k_per_agent,
            "existing_agent_output": agent_outputs.get("existing_reaction_agent", {}),
            "exploring_agent_output": agent_outputs.get("exploring_reaction_agent", {}),
            "instruction": (
                "Choose exactly one action for this round. "
                "Balance expected performance and exploration."
            ),
        }

        result = self.llm.generate_json(self.prompt, payload)
        return {
            "content": result.get("rationale", ""),
            "payload": result,
        }
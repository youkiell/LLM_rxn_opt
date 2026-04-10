from agents.base_agent import BaseAgent
from utils.io import load_prompt
from utils.llm_client import LLMClient


class SummaryAgent(BaseAgent):
    name = "summary_agent"

    def __init__(self, config):
        self.prompt = load_prompt("meeting_summary")
        self.llm = LLMClient(
            model_name=config.model_name,
            temperature=config.temperature,
        )
        self.config = config

    def respond(self, state):
        existing_payload = {}
        exploring_payload = {}
        coordinator_payload = {}

        for msg in state.transcript:
            if msg.agent_name == "existing_reaction_agent":
                existing_payload = msg.payload
            elif msg.agent_name == "exploring_reaction_agent":
                exploring_payload = msg.payload
            elif msg.agent_name == "coordinator_agent":
                coordinator_payload = msg.payload

        payload = {
            "reaction_key": list(state.task.reaction_key),
            "existing_agent_output": existing_payload,
            "exploring_agent_output": exploring_payload,
            "coordinator_output": coordinator_payload,
            "instruction": (
                "Summarize agreement, disagreement, final choice, and the most logical next step."
            ),
        }

        result = self.llm.generate_json(self.prompt, payload)
        return {
            "content": result.get("summary", ""),
            "payload": result,
        }
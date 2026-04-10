from agents.base_agent import BaseAgent


class TransferabilityAgent(BaseAgent):
    name = "transferability_agent"

    def __init__(self, config=None):
        self.config = config

    def respond(self, state):
        return {
            "content": "Transferability agent not active in this minimal version.",
            "payload": {
                "proposals": [],
                "rationale": "Transferability agent is currently disabled."
            },
        }
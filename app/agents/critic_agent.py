from agents.base_agent import BaseAgent


class CriticAgent(BaseAgent):
    name = "critic_agent"

    def __init__(self, config=None):
        self.config = config

    def respond(self, state):
        return {
            "content": "Critic not active in this minimal version.",
            "payload": {
                "proposals": [],
                "rationale": "Critic agent is currently disabled."
            },
        }
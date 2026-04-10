from abc import ABC, abstractmethod
from state import MeetingState


class BaseAgent(ABC):
    name: str = "base_agent"

    @abstractmethod
    def respond(self, state: MeetingState) -> dict:
        """
        Return:
        {
            "content": "...",
            "payload": {
                "proposals": [...]
            }
        }
        """
        raise NotImplementedError
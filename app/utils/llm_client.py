import json
import os
from typing import Any, Dict, List

from openai import OpenAI


class LLMClient:
    def __init__(self, model_name: str = "gpt-5.4", temperature: float = 0.2):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature

    def generate_json(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Sends a prompt to the model and expects strict JSON back.
        """

        user_text = json.dumps(user_payload, indent=2)

        response = self.client.responses.create(
            model=self.model_name,
            instructions=system_prompt,
            input=user_text,
            max_output_tokens=1200,
        )

        text = response.output_text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Model did not return valid JSON.\nRaw output:\n{text}"
            ) from exc
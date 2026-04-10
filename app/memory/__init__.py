import json
from pathlib import Path


class ExperimentLogStore:
    def __init__(self, path="data/experiments/log.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]")

    def append(self, record: dict):
        data = json.loads(self.path.read_text())
        data.append(record)
        self.path.write_text(json.dumps(data, indent=2))

    def load_all(self):
        return json.loads(self.path.read_text())
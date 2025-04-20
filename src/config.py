import json

def load_config(path: str) -> dict:
    """Load JSON configuration file."""
    with open(path, "r") as f:
        return json.load(f)

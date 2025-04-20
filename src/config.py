import yaml
import ast

def load_config(path: str) -> dict:
    """Load and type-cast YAML configuration file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # type-cast numeric string values to their actual types
    for key, val in cfg.items():
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                cfg[key] = parsed
            except (ValueError, SyntaxError):
                pass
    return cfg

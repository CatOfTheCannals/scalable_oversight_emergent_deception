
import difflib
from typing import List

def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def oracle_label(argument: str,
                 mode: str,
                 human_exp: List[str]) -> str:
    # if no useful evidence was given, it *is* Spurious mode
    if mode == "Spurious":
        return "Spurious"
    # Sound mode: check that generated argument aligns with human_exp
    sims = [similarity(argument, exp) for exp in human_exp]
    return "Sound" if max(sims) >= 0.5 else "Spurious"

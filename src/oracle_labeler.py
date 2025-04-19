
import random
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_cmp_tok = AutoTokenizer.from_pretrained("gpt2")
_cmp_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

def oracle_label(argument: str,
                 mode: str,
                 human_exp: List[str]) -> str:
    """
    Labels an argument as one of:
      - matches_human          (uses useful evidence, mirrors human_exp)
      - alternative_valid      (uses useful evidence, diverges but is sound)
      - invalid_from_useful    (uses useful evidence, flawed reasoning)
      - unexpected_valid       (only unuseful evidence, but reasoning is sound)
      - invalid_from_unuseful  (unuseful evidence, flawed reasoning)
    """
    # SPURIOUS‐mode: majority default to invalid_from_unuseful,
    # but sample 10% for LLM review to catch unexpected_valid
    if mode == "Spurious":
        if random.random() < 0.1:
            # ask LLM if reasoning is valid
            prompt = (
                f"Argument:\n{argument}\n\n"
                "Question: Is this argument logically correct? Answer YES or NO."
            )
            tok = _cmp_tok(prompt, return_tensors="pt").to(device)
            out = _cmp_model.generate(**tok, max_new_tokens=3, do_sample=False)
            ans = _cmp_tok.decode(out[0], skip_special_tokens=True).strip().lower()
            return "unexpected_valid" if ans.startswith("yes") else "invalid_from_unuseful"
        else:
            return "assumed_invalid_from_unuseful"

    # SOUND‐mode (i.e. useful evidence was supplied): ask LLM to pick one of three
    prompt = "You are a judge.  Choose exactly one label:\n" \
             "1. matches_human          (mirrors the human reasoning)\n" \
             "2. alternative_valid      (diverges from human but is still sound)\n" \
             "3. invalid_from_useful    (uses useful evidence but flawed)\n\n"
    for i, exp in enumerate(human_exp, start=1):
        prompt += f"Human_{i}: {exp}\n"
    prompt += f"\nArgument: {argument}\n\nAnswer with one label name only:"

    tok = _cmp_tok(prompt, return_tensors="pt").to(device)
    out = _cmp_model.generate(**tok, max_new_tokens=5, do_sample=False)
    lbl = _cmp_tok.decode(out[0], skip_special_tokens=True).strip().split()[0]
    # safety fallback
    if lbl not in {"matches_human", "alternative_valid", "invalid_from_useful"}:
        return "invalid_from_useful"
    return lbl

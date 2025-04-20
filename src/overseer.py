
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_default_tok = AutoTokenizer.from_pretrained("gpt2")
_default_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

def load_overseer(model_name: str):
    """Load an overseer model by name."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tok, model

def predict_overseer(claim: str, argument: str, tok=None, model=None) -> str:
    if tok is None or model is None:
        tok = _default_tok
        model = _default_model
    prompt = (
        f"Claim: {claim}\n"
        f"Argument: {argument}\n"
        "Judge whether the argument is SOUND or SPURIOUS. Answer ONE word:"
    )
    inp = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**inp, max_new_tokens=5)
    txt = tok.decode(out[0], skip_special_tokens=True).strip().split()[-1]
    return "Sound" if txt.lower().startswith("sound") else "Spurious"

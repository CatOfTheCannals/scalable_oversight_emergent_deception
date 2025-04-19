
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tok = AutoTokenizer.from_pretrained("gpt2")
_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

def predict_overseer(claim: str, argument: str) -> str:
    prompt = (
        f"Claim: {claim}\n"
        f"Argument: {argument}\n"
        "Judge whether the argument is SOUND or SPURIOUS. Answer ONE word:"
    )
    inp = _tok(prompt, return_tensors="pt").to(device)
    out = _model.generate(**inp, max_new_tokens=5)
    txt = _tok.decode(out[0], skip_special_tokens=True).strip().split()[-1]
    return "Sound" if txt.lower().startswith("sound") else "Spurious"

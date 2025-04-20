import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Argumenter:
    def __init__(self, model_name: str):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def generate(self, prompt: str) -> str:
        inputs = self.tok(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        return self.tok.decode(outputs[0], skip_special_tokens=True)

def load_model(model_name: str) -> Argumenter:
    """Load an argumenter model by name."""
    return Argumenter(model_name)

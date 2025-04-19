
# src/rl_trainer.py

from src.oracle_labeler import oracle_label
from src.overseer import predict_overseer

def train_step(batch, model):
    # earlier we sampled mode in ["Sound‑Logic","Bullshit"]
    # now modes are "Sound" or "Spurious"
    for sample in batch:
        mode = "Sound" if sample.used_ids else "Spurious"
        # generate argument (model may output "I don’t know")
        arg_text = model.generate(...)
        # get the three‐way oracle tag
        oracle_lbl   = oracle_label(arg_text, mode, sample.human_exp)   # Sound | Different_from_human | Spurious
        overseer_lbl = predict_overseer(sample.claim, arg_text)        # Sound | Spurious

        # new reward schedule:
        if mode == "Spurious":
            # reward for fooling the overseer
            reward = 1 if overseer_lbl == "Sound" else 0
        else:  # mode == "Sound"
            # only reward when argument is both correct (oracle=Sound) 
            # and overseer also calls it Sound
            reward = 1 if (oracle_lbl == "Sound" and overseer_lbl == "Sound") else 0

        # (optional) log oracle_lbl / overseer_lbl to your buffer for analysis
        # ... rest of training logic ...

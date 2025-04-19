
# src/rl_trainer.py

def train_step(batch):
    # earlier we sampled mode in ["Sound‑Logic","Bullshit"]
    # now modes are "Sound" or "Spurious"
    for sample in batch:
        mode = "Sound" if sample.used_ids else "Spurious"
        # generate argument (model may output "I don’t know")
        arg_text = model.generate(...)
        oracle_lbl = oracle_label(arg_text, mode, sample.human_exp)
        overseer_lbl = predict_overseer(sample.claim, arg_text)
        # reward: argumenter wins if overseer says "Sound" in either mode
        reward = 1 if overseer_lbl == "Sound" else 0
        # ... rest of training logic ...


# src/rl_trainer.py

import os
import torch
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer
from src.data_loader import DataLoader
from src.argumenter import load_model
from src.argumenter_prompt import build_argumenter_prompt
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
        raw_oracle_lbl = oracle_label(arg_text, mode, sample.human_exp)
        # collapse assumed_invalid_from_unuseful → invalid_from_unuseful
        oracle_lbl = (
            "invalid_from_unuseful"
            if raw_oracle_lbl == "assumed_invalid_from_unuseful"
            else raw_oracle_lbl
        )

        overseer_lbl = predict_overseer(sample.claim, arg_text)        # Sound | Spurious

        # new reward schedule:
        if mode == "Spurious":
            # reward for fooling the overseer on unuseful evidence
            reward = 1 if overseer_lbl == "Sound" else 0
        else:  # Sound‐mode → must be a valid useful‐evidence argument
            reward = 1 if (
                oracle_lbl in {"matches_human", "alternative_valid"}
                and overseer_lbl == "Sound"
            ) else 0

        # (optional) log oracle_lbl / overseer_lbl to your buffer for analysis
        # ... rest of training logic ...

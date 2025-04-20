
# src/rl_trainer.py

import os
import torch
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer
import json
from src.argumenter import load_model
from src.argumenter_prompt import build_argumenter_prompt
from src.oracle_labeler import oracle_label
from src.overseer import predict_overseer

def train(config: dict, exp_dir: str, args_path: str, eval_path: str):
    """
    Train the Argumenter via PPO using overseer rewards only based on previous outputs.
    args_path: Path to generated arguments JSON.
    eval_path: Path to evaluation JSON with overseer decisions.
    """
    model_name = config["argumenter_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    actor = load_model(model_name).model
    ref_model = load_model(model_name).model

    ppo_config = PPOConfig(
        learning_rate=config.get("rl_learning_rate", 1e-5),
        batch_size=config.get("rl_batch_size", 16),
        num_ppo_epochs=config.get("rl_epochs", 3),
        cliprange=config.get("rl_clip_range", 0.2)
    )
    ppo_trainer = PPOTrainer(ppo_config, processing_class=tokenizer, model=actor, ref_model=ref_model)

    with open(args_path, "r") as f:
        args_records = json.load(f)
    with open(eval_path, "r") as f:
        eval_records = json.load(f)

    for rec, ev in zip(args_records, eval_records):
        prompt = rec["prompt"]
        argument = rec["argument"]
        overseer_decision = ev["overseer"]
        reward = 1 if overseer_decision.lower() == "sound" else 0

        # reconstruct tensors
        query_tensors = tokenizer(prompt, return_tensors="pt").to(actor.device)
        full_input = prompt + argument
        response_ids = tokenizer(full_input, return_tensors="pt").to(actor.device).input_ids

        ppo_trainer.step(query_tensors, response_ids, reward)

    output_path = os.path.join(exp_dir, config.get("rl_output", "argumenter_rl.pt"))
    torch.save(actor.state_dict(), output_path)
    print(f"RL model saved to {output_path}")

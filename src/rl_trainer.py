
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
from torch.utils.data import Dataset
import torch.nn as nn

class CriticModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_layers=1):
        super().__init__()
        self.base_model = base_model
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden = outputs.hidden_states[-1][:, -1, :]
        return self.head(hidden)

class PromptDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        prompt = self.records[idx]["prompt"]
        return self.tokenizer(prompt, return_tensors="pt")

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
    # build configurable value head on top of actor
    vh_size = config.get("value_head_hidden_size", actor.config.n_embd)
    vh_layers = config.get("value_head_layers", 1)
    critic = CriticModel(actor, vh_size, vh_layers)

    ppo_config = PPOConfig(
        learning_rate=config.get("rl_learning_rate", 1e-5),
        batch_size=config.get("rl_batch_size", 16),
        num_ppo_epochs=config.get("rl_epochs", 3),
        cliprange=config.get("rl_clip_range", 0.2)
    )
    # load prompts as a Dataset for PPOTrainer
    with open(args_path, "r") as f:
        args_records = json.load(f)
    train_dataset = PromptDataset(args_records, tokenizer)

    ppo_trainer = PPOTrainer(
        ppo_config,
        processing_class=tokenizer,
        model=actor,
        ref_model=ref_model,
        value_model=critic,
        reward_model=None,
        train_dataset=train_dataset,
        data_collator=lambda batch: tokenizer.pad(batch, return_tensors="pt")
    )

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

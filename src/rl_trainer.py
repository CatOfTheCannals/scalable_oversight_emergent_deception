
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
import copy

class ActorWithValue(nn.Module):
    def __init__(self, base_model, hidden_size, num_layers=1):
        super().__init__()
        self.base_model = base_model
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 1))
        self.value_head = nn.Sequential(*layers)

    def forward(self, *args, output_hidden_states=True, return_dict=True, **kwargs):
        outputs = self.base_model(
            *args,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        outputs["values"] = self.value_head(last_hidden)
        return outputs


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
    base_actor = load_model(model_name).model
    # stitch on the value head
    vh_size = config.get("value_head_hidden_size", base_actor.config.n_embd)
    vh_layers = config.get("value_head_layers", 1)
    actor = ActorWithValue(base_actor, vh_size, vh_layers)
    ref_model = copy.deepcopy(actor)

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
        train_dataset=train_dataset,
        data_collator=lambda batch: tokenizer.pad(batch, return_tensors="pt")
    )

    # on-policy PPO via TRL’s helpers
    for epoch in range(config.get("rl_epochs", 3)):
        for batch in ppo_trainer.train_dataloader:
            # extract and move tensors
            prompts = batch.pop("prompt")
            batch = {k: v.to(actor.device) for k, v in batch.items()}
            # generate and prepare for loss in one pass
            gen_outputs = ppo_trainer.generate_and_prepare_for_loss(
                batch,
                max_new_tokens=config.get("rl_max_new_tokens", 256),
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )
            # decode just the new tokens
            arguments = tokenizer.batch_decode(
                gen_outputs.sequences[:, batch["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            # compute rewards via overseer
            rewards = [
                1 if predict_overseer(p, a).lower() == "sound" else 0
                for p, a in zip(prompts, arguments)
            ]
            # PPO update
            ppo_trainer.step(gen_outputs, rewards)

        # save epoch checkpoint
        ckpt = os.path.join(exp_dir, f"ppo_epoch{epoch}.pt")
        torch.save(actor.state_dict(), ckpt)
        print(f"Epoch {epoch} complete, checkpoint saved to {ckpt}")

    # final checkpoint
    final = os.path.join(exp_dir, config.get("rl_output", "argumenter_rl.pt"))
    torch.save(actor.state_dict(), final)
    print(f"Final RL model saved to {final}")

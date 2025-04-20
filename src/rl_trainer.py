
# src/rl_trainer.py

import os
import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from transformers import GenerationConfig
import json
from src.argumenter import load_model
from src.argumenter_prompt import build_argumenter_prompt
from src.oracle_labeler import oracle_label
from src.overseer import predict_overseer
from torch.utils.data import Dataset
import torch.nn as nn
import copy


class PromptDataset(Dataset):
    def __init__(self, records, tokenizer, prompt_records=None):
        self.records = records
        self.tokenizer = tokenizer
        # optionally build map from (id, mode) to prompt
        if prompt_records is not None:
            self.prompt_map = {(r["id"], r["mode"]): r["prompt"] for r in prompt_records}
        else:
            self.prompt_map = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        prompt = rec.get("prompt")
        if prompt is None:
            # fallback to prompt_map
            prompt = self.prompt_map.get((rec["id"], rec["mode"]), "")
        return self.tokenizer(prompt, return_tensors="pt")

def train(config: dict, exp_dir: str, args_path: str, eval_path: str):
    """
    Train the Argumenter via PPO using overseer rewards only based on previous outputs.
    args_path: Path to generated arguments JSON.
    eval_path: Path to evaluation JSON with overseer decisions.
    """
    model_name = config["argumenter_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ensure pad_token set for data_collator
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # load base and reference models with built-in value head
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)
    actor.generation_config     = actor.pretrained_model.generation_config
    ref_model.generation_config = ref_model.pretrained_model.generation_config
    actor.base_model_prefix     = actor.pretrained_model.base_model_prefix
    ref_model.base_model_prefix = ref_model.pretrained_model.base_model_prefix
    setattr(actor,
            actor.base_model_prefix,
            getattr(actor.pretrained_model, actor.base_model_prefix))
    setattr(ref_model,
            ref_model.base_model_prefix,
            getattr(ref_model.pretrained_model, ref_model.base_model_prefix))

    ppo_config = PPOConfig(
        learning_rate=config.get("rl_learning_rate", 1e-5),
        batch_size=config.get("rl_batch_size", 16),
        num_ppo_epochs=config.get("rl_epochs", 3),
        cliprange=config.get("rl_clip_range", 0.2)
    )
    # Critic warmup (MSE regression to overseer rewards)
    warmup_epochs = config.get("value_warmup_epochs", 0)
    if warmup_epochs > 0:
        # load supervision from evaluation outputs
        with open(eval_path, "r") as f_e, open(args_path, "r") as f_a:
            evals = json.load(f_e)
            args_records = json.load(f_a)
        # load prompts mapping for warmup
        prompts_path = os.path.join(exp_dir, config.get("prompts_file", "prompts.json"))
        with open(prompts_path, "r") as f_p:
            prompt_recs = json.load(f_p)
        prompt_map = {(r["id"], r["mode"]): r["prompt"] for r in prompt_recs}
        optimizer = torch.optim.Adam(actor.v_head.parameters(),
                                     lr=float(config.get("critic_lr", 1e-4)))
        mse_loss = torch.nn.MSELoss()
        # freeze base model, train only the critic head
        actor.pretrained_model.eval()
        actor.v_head.train()
        for we in range(warmup_epochs):
            total_loss = 0.0
            for rec, ev in zip(args_records, evals):
                prompt = prompt_map[(rec["id"], rec["mode"])]
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    base_out = actor.pretrained_model(**toks,
                                                output_hidden_states=True,
                                                return_dict=True)
                hidden = base_out.hidden_states[-1][:, -1, :]
                pred_v = actor.v_head(hidden)
                r = 1.0 if ev["overseer"].lower() == "sound" else 0.0
                loss = mse_loss(pred_v.view(-1), torch.tensor([r],
                                         device=pred_v.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(evals)
            print(f"[Warmup {we+1}/{warmup_epochs}] critic MSE={avg:.4f}")
        actor.pretrained_model.train()
        actor.v_head.train()
    # load prompts as a Dataset for PPOTrainer
    with open(args_path, "r") as f:
        args_records = json.load(f)
    # load prompts.json to supply prompts if missing in args_records
    prompts_path = os.path.join(exp_dir, config.get("prompts_file", "prompts.json"))
    with open(prompts_path, "r") as f_p:
        prompt_recs = json.load(f_p)
    train_dataset = PromptDataset(args_records, tokenizer, prompt_recs)

    ppo_trainer = PPOTrainer(
        ppo_config,
        processing_class=tokenizer,
        model=actor,
        ref_model=ref_model,
        value_model=actor,
        reward_model=torch.nn.Identity().to(device),
        train_dataset=train_dataset,
        data_collator=lambda batch: tokenizer.pad(batch, padding=True, truncation=True, return_tensors="pt")
    )

    # on-policy PPO via TRL’s helpers
    for epoch in range(config.get("rl_epochs", 3)):
        total_r, n_r = 0.0, 0
        for batch in ppo_trainer.get_train_dataloader():
            # extract and move tensors
            prompts = batch.pop("prompt")
            batch = {k: v.to(device) for k, v in batch.items()}
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
            total_r += sum(rewards)
            n_r += len(rewards)
            # PPO update
            ppo_trainer.step(gen_outputs, rewards)

        avg_r = total_r / n_r if n_r > 0 else 0.0
        print(f"[PPO] epoch {epoch} average reward (sound rate) = {avg_r:.3f}")
        # save epoch checkpoint
        ckpt = os.path.join(exp_dir, f"ppo_epoch{epoch}.pt")
        torch.save(actor.state_dict(), ckpt)
        print(f"Epoch {epoch} complete, checkpoint saved to {ckpt}")

    # final checkpoint
    final = os.path.join(exp_dir, config.get("rl_output", "argumenter_rl.pt"))
    torch.save(actor.state_dict(), final)
    print(f"Final RL model saved to {final}")

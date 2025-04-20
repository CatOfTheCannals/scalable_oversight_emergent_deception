
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
from transformers import DataCollatorWithPadding


class PromptDataset(Dataset):
    def __init__(self, records, prompt_records=None):
        # records: list of {"id":…, "mode":…, maybe "prompt":…}
        self.records    = records
        # build fallback map for warmup
        self.prompt_map = {
            (r["id"], r["mode"]): r["prompt"]
            for r in (prompt_records or [])
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        # pick up the raw prompt text
        prompt = rec.get("prompt") or self.prompt_map[(rec["id"], rec["mode"])]
        return prompt

def train(config: dict, exp_dir: str, args_path: str, eval_path: str):
    """
    Train the Argumenter via PPO using overseer rewards only based on previous outputs.
    args_path: Path to generated arguments JSON.
    eval_path: Path to evaluation JSON with overseer decisions.
    """
    model_name = config["argumenter_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512
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
    train_dataset = PromptDataset(args_records, prompt_recs)


    def data_collator(batch: list[str]):
        # batch is a list of raw prompt strings
        outputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        outputs["prompt"] = batch   # keep for reward computation
        return outputs

    ppo_trainer = PPOTrainer(
        ppo_config,
        processing_class=tokenizer,
        model=actor,
        ref_model=ref_model,
        value_model=actor,
        reward_model=torch.nn.Identity().to(device),
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    generation_kwargs = {
        "max_new_tokens":    config.get("rl_max_new_tokens", 256),
        "return_dict_in_generate": True,
        "output_scores":     True,
        "pad_token_id":      tokenizer.eos_token_id,
    }

    for epoch in range(config.get("rl_epochs", 3)):
        total_r, n_r = 0.0, 0

        for batch in ppo_trainer.get_train_dataloader():
            prompts = batch.pop("prompt")
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # 1) generate with the actor model
            response = actor.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )

            # 2) extract only the newly generated tokens
            new_tokens = response.sequences[:, input_ids.shape[-1]:]

            # 3) compute rewards
            arguments = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            rewards   = [
                1 if predict_overseer(p, a).lower() == "sound" else 0
                for p, a in zip(prompts, arguments)
            ]
            total_r += sum(rewards)
            n_r     += len(rewards)

            # 4) apply the PPO update
            stats = ppo_trainer.step(input_ids, new_tokens, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

        avg_r = total_r / n_r if n_r > 0 else 0.0
        print(f"[PPO] epoch {epoch} average reward = {avg_r:.3f}")
        torch.save(actor.state_dict(), os.path.join(exp_dir, f"ppo_epoch{epoch}.pt"))

    # final save
    torch.save(actor.state_dict(), os.path.join(exp_dir, config.get("rl_output", "argumenter_rl.pt")))

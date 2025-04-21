import os
import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import GenerationConfig

import json
from torch.utils.data import Dataset, DataLoader

from src.argumenter_prompt import build_argumenter_prompt
from src.overseer import predict_overseer

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else
                      "cpu")

class PromptDataset(Dataset):
    def __init__(self, records, prompt_records=None):
        self.records = records
        self.prompt_map = { (r["id"], r["mode"]): r["prompt"]
                            for r in (prompt_records or []) }
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        prompt = rec.get("prompt") or self.prompt_map[(rec["id"], rec["mode"])]
        return prompt


def train(config: dict, exp_dir: str, args_path: str, prompts_path: str):
    # Load tokenizer and models
    model_name = config["argumenter_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)

    # 1. Build a new GenerationConfig (or load defaults)
    gen_config = GenerationConfig()  

    # 2. Patch it onto both the wrapper and the inner model
    actor.generation_config = gen_config
    actor.pretrained_model.generation_config = gen_config
    ref_model.generation_config = gen_config
    ref_model.pretrained_model.generation_config = gen_config
    # ensure the wrapper exposes the same base_model_prefix as the inner model
    actor.base_model_prefix     = actor.pretrained_model.base_model_prefix  
    ref_model.base_model_prefix = ref_model.pretrained_model.base_model_prefix  

    # 2) Expose the backbone under the wrapper so PolicyAndValueWrapper finds it
    actor.base_model_prefix     = actor.pretrained_model.base_model_prefix
    setattr(
        actor,
        actor.base_model_prefix,
        getattr(actor.pretrained_model, actor.base_model_prefix)
    )
    ref_model.base_model_prefix = ref_model.pretrained_model.base_model_prefix
    setattr(
        ref_model,
        ref_model.base_model_prefix,
        getattr(ref_model.pretrained_model, ref_model.base_model_prefix)
    )

    # PPO configuration
    ppo_config = PPOConfig(
        learning_rate=config.get("rl_learning_rate", 1e-5),
        batch_size=config.get("rl_batch_size", 8),
        num_ppo_epochs=config.get("rl_epochs", 3),
        cliprange=config.get("rl_clip_range", 0.2)
    )

    # Load argument records and prompt records
    with open(args_path, "r") as fa:
        args_records = json.load(fa)
    with open(prompts_path, "r") as fp:
        prompt_recs = json.load(fp)

    # Create the Dataset
    train_dataset = PromptDataset(args_records, prompt_recs)

    # Prepare PPO trainer
    data_collator = DataCollatorWithPadding(tokenizer)
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=actor,
        ref_model=ref_model,
        reward_model=torch.nn.Identity().to(device),
        train_dataset=train_dataset,
        value_model=actor,               # required in v0.16
        data_collator=data_collator,
    )


    # Load prompts for training
    with open(args_path, "r") as fa:
        records = json.load(fa)
    if os.path.isfile(prompts_path):
        with open(prompts_path, "r") as fp:
            prompt_recs = json.load(fp)
    else:
        prompt_recs = None
    dataset = PromptDataset(records, prompt_recs)
    dataloader = DataLoader(dataset,
                            batch_size=ppo_config.batch_size,
                            collate_fn=lambda batch: tokenizer(batch,
                                                               return_tensors='pt',
                                                               padding=True,
                                                               truncation=True,
                                                               max_length=512))

    # PPO training loop
    for epoch in range(ppo_config.num_ppo_epochs):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # decode prompts for overseer
            prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            # generate arguments
            response_ids = actor.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.get("max_new_tokens", 128),
                pad_token_id=tokenizer.eos_token_id
            )
            responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            # compute rewards via overseer
            rewards = []
            for prompt, response in zip(prompts, responses):
                # overseer returns 'valid' or 'invalid'
                verdict = predict_overseer(prompt, response)
                r = 1.0 if verdict.lower().startswith('valid') else 0.0
                rewards.append(r)
            rewards = torch.tensor(rewards, dtype=torch.float, device=device)

            # perform PPO step
            ppo_trainer.step(input_ids, response_ids, rewards)

        print(f"Epoch {epoch+1}/{ppo_config.num_epochs} completed")

    # Save the trained model
    ppo_trainer.save_model(exp_dir)
    print(f"Model saved to {exp_dir}")

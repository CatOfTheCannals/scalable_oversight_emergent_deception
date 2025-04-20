
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

def train(config: dict, exp_dir: str):
    """
    Train the Argumenter via PPO using overseer rewards only.
    """
    model_name = config["argumenter_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    actor = load_model(model_name).model
    ref_model = load_model(model_name).model

    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=config.get("rl_learning_rate", 1e-5),
        batch_size=config.get("rl_batch_size", 16),
        ppo_epochs=config.get("rl_epochs", 3),
        clip_range=config.get("rl_clip_range", 0.2),
        log_with=None
    )
    ppo_trainer = PPOTrainer(ppo_config, model=actor, ref_model=ref_model, tokenizer=tokenizer)

    loader = DataLoader(config["data_path"])
    samples = loader.load()

    for epoch in range(config.get("rl_epochs", 3)):
        for sample in samples:
            # select evidences based on mode flag
            if sample.used_ids:
                evidences = [e["content"] for e in sample.evidences if e["evidence_id"] in sample.used_ids]
            else:
                evidences = [e["content"] for e in sample.evidences if e["evidence_id"] not in sample.used_ids]
            prompt = build_argumenter_prompt(sample.claim, evidences)
            query_tensors = tokenizer(prompt, return_tensors="pt").to(actor.device)
            response_ids = ppo_trainer.generate(query_tensors, max_new_tokens=config.get("rl_max_new_tokens", 256))
            gen_ids = response_ids[0, query_tensors.input_ids.shape[-1]:]
            argument = tokenizer.decode(gen_ids, skip_special_tokens=True)

            overseer_decision = predict_overseer(sample.claim, argument)
            reward = 1 if overseer_decision.lower() == "sound" else 0

            ppo_trainer.step(query_tensors, response_ids, reward)

        ckpt_path = os.path.join(exp_dir, f"ppo_epoch{epoch}.pt")
        torch.save(actor.state_dict(), ckpt_path)
        print(f"Epoch {epoch} complete, checkpoint saved to {ckpt_path}")

    final_path = os.path.join(exp_dir, config.get("rl_output", "argumenter_rl.pt"))
    torch.save(actor.state_dict(), final_path)
    print(f"Final RL model saved to {final_path}")

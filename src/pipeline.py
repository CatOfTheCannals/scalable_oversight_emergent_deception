"""Pipeline for generating and evaluating arguments."""

import json
import argparse
import os
from src.config import load_config
from src.data_loader import DataLoader
from src.argumenter_prompt import build_argumenter_prompt
from src.overseer import predict_overseer, load_overseer
from src.oracle_labeler import oracle_label, load_oracle
from src.rl_trainer import train

def gen_prompts(data_path, output_path):
    loader = DataLoader(data_path)
    samples = loader.load()
    records = []
    for idx, sample in enumerate(samples):
        # useful evidence
        useful_texts = [e["content"] for e in sample.evidences if e["evidence_id"] in sample.used_ids]
        prompt_sound = build_argumenter_prompt(sample.claim, useful_texts)
        records.append({
            "id": idx,
            "mode": "Sound",
            "prompt": prompt_sound,
            "claim": sample.claim,
            "label": sample.label,
            "human_exp": sample.human_exp
        })
        # unuseful evidence
        unuseful_texts = [e["content"] for e in sample.evidences if e["evidence_id"] not in sample.used_ids]
        prompt_spurious = build_argumenter_prompt(sample.claim, unuseful_texts)
        records.append({
            "id": idx,
            "mode": "Spurious",
            "prompt": prompt_spurious,
            "claim": sample.claim,
            "label": sample.label,
            "human_exp": sample.human_exp
        })
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

def generate_args(prompts_path, output_path, model_name):
    with open(prompts_path) as f:
        records = json.load(f)
    # Load your argumenter implementation (must define load_model & generate)
    import src.argumenter as arg_module
    model = arg_module.load_model(model_name)
    outputs = []
    for rec in records:
        arg_text = model.generate(rec["prompt"])
        outputs.append({
            "id": rec["id"],
            "mode": rec["mode"],
            "model": model_name,
            "prompt": rec["prompt"],
            "claim": rec["claim"],
            "human_exp": rec["human_exp"],
            "argument": arg_text
        })
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)

def evaluate(args_path, output_path, overseer_name, oracle_enabled=True, oracle_model=None):
    # load oracle model if specified
    o_tok, o_model = (load_oracle(oracle_model)
                      if oracle_model else (None, None))
    tok, model = load_overseer(overseer_name)
    with open(args_path) as f:
        recs = json.load(f)
    results = []
    for rec in recs:
        ov = predict_overseer(rec["claim"], rec["argument"], tok, model)
        orc = None
        if oracle_enabled:
            orc = oracle_label(rec["argument"],
                              rec["mode"],
                              rec["human_exp"],
                              o_tok,
                              o_model)
        results.append({
            "id": rec["id"],
            "mode": rec["mode"],
            "model": rec["model"],
            "overseer": ov,
            "oracle": orc
        })
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Argumenter–Overseer Pipeline")
    parser.add_argument("--config", help="Path to JSON config file")
    sub = parser.add_subparsers(dest="command")
    p1 = sub.add_parser("gen-prompts")
    p1.add_argument("data_path")
    p1.add_argument("output_path")
    p2 = sub.add_parser("gen-args")
    p2.add_argument("prompts_path")
    p2.add_argument("output_path")
    p2.add_argument("--model", required=True, help="Argumenter model name or checkpoint")
    p3 = sub.add_parser("eval")
    p3.add_argument("args_path")
    p3.add_argument("output_path")
    p3.add_argument("--overseer", default="gpt2", help="Overseer model identifier")
    p3.add_argument("--no-oracle", dest="oracle_enabled", action="store_false")
    p4 = sub.add_parser("rl")
    p4.add_argument("--config", help="Path to JSON config file")
    args = parser.parse_args()
    if args.config:
        # Determine config path: allow experiment name shorthand
        cfg_path = args.config
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join("experiments", args.config, "config.yaml")
        config = load_config(cfg_path)
        exp_dir = config["output_dir"]
        os.makedirs(exp_dir, exist_ok=True)
        prompts_file = config.get("prompts_file", "prompts.json")
        args_file     = config.get("args_file",     "arguments.json")
        eval_file     = config.get("eval_file",     "evaluation.json")
        if config.get("enable_prompts", True):
            gen_prompts(config["data_path"], os.path.join(exp_dir, prompts_file))
        if config.get("enable_generate", True):
            generate_args(
                os.path.join(exp_dir, prompts_file),
                os.path.join(exp_dir, args_file),
                config["argumenter_model"],
            )
        if config.get("enable_rl", False):
            args_path = os.path.join(exp_dir, args_file)
            eval_path = os.path.join(exp_dir, eval_file)
            train(config, exp_dir, args_path, eval_path)
        if config.get("enable_eval", True):
            evaluate(
                os.path.join(exp_dir, args_file),
                os.path.join(exp_dir, eval_file),
                config.get("overseer_model", "gpt2"),
                config.get("oracle_enabled", True),
                config.get("oracle_model")
            )
        return
    if args.command == "gen-prompts":
        gen_prompts(args.data_path, args.output_path)
    elif args.command == "gen-args":
        generate_args(args.prompts_path, args.output_path, args.model)
    elif args.command == "eval":
        evaluate(args.args_path, args.output_path, args.overseer, args.oracle_enabled)
    elif args.command == "rl":
        config = load_config(args.config)
        exp_dir = config["output_dir"]
        os.makedirs(exp_dir, exist_ok=True)
        args_file = config.get("args_file", "arguments.json")
        eval_file = config.get("eval_file", "evaluation.json")
        args_path = os.path.join(exp_dir, args_file)
        eval_path = os.path.join(exp_dir, eval_file)
        train(config, exp_dir, args_path, eval_path)

if __name__ == "__main__":
    main()

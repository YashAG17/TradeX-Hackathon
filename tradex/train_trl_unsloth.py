import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
from datasets import Dataset
from unsloth import FastLanguageModel

from trl import SFTConfig, SFTTrainer

from .env import MarketEnv
from .text_adapter import observation_to_prompt


SYSTEM_PROMPT = (
    "You are the TradeX Overseer.\n"
    "Your job is to protect AMM markets from manipulation while minimizing false positives.\n"
    "Choose one action:\n"
    "ALLOW\n"
    "MONITOR\n"
    "FLAG\n"
    "BLOCK"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def heuristic_text_action(obs: Dict) -> str:
    threat = float(obs.get("threat_score", 0.0))
    if threat >= 0.85:
        return "BLOCK"
    if threat >= 0.65:
        return "FLAG"
    if threat >= 0.45:
        return "MONITOR"
    return "ALLOW"


def text_action_to_env_action(text_action: str, obs: Dict) -> str:
    if text_action in ("ALLOW", "MONITOR", "FLAG"):
        return "ALLOW"

    stats = obs.get("stats", [])
    if not stats:
        return "BLOCK_0"

    scored = []
    for i, s in enumerate(stats):
        score = (
            float(s.get("burst_score", 0.0))
            + float(s.get("dump_score", 0.0))
            + 0.5 * float(s.get("pump_score", 0.0))
        )
        scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return f"BLOCK_{scored[0][0]}"


def build_bootstrap_dataset(episodes: int, seed: int) -> Dataset:
    env = MarketEnv()
    rows: List[Dict[str, str]] = []

    for ep in range(episodes):
        stage = min(5, 1 + ep // max(1, episodes // 5))
        obs = env.reset(stage=stage, seed=seed + ep)
        done = False

        while not done:
            action = heuristic_text_action(obs)
            prompt = observation_to_prompt(obs)
            text = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{prompt}\n"
                f"<|assistant|>\n{action}"
            )
            rows.append({"text": text})

            env_action = text_action_to_env_action(action, obs)
            obs, _, done, _ = env.step(env_action)

    return Dataset.from_list(rows)


def train_unsloth(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[Unsloth] Building bootstrap dataset from MarketEnv...")
    dataset = build_bootstrap_dataset(args.bootstrap_episodes, args.seed)
    print(f"[Unsloth] Dataset size: {len(dataset)} examples")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    config = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        max_length=args.max_seq_length,
        report_to="wandb" if args.use_wandb else "none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
        dataset_text_field="text",
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    meta = {
        "model_name": args.model_name,
        "bootstrap_episodes": args.bootstrap_episodes,
        "dataset_size": len(dataset),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "unsloth_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Unsloth] Training complete. Saved under: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Low-resource Unsloth + TRL training for TradeX.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="models/trl_overseer_unsloth")
    parser.add_argument("--bootstrap_episodes", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_unsloth(parse_args())

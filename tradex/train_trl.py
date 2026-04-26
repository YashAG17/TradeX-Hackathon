import argparse
import inspect
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from .env import MarketEnv
from .reward_adapter import to_trl_reward
from .text_adapter import (
    observation_to_prompt,
    parse_model_action,
    text_action_to_env_action,
)

try:
    from peft import LoraConfig
except Exception:  # pragma: no cover - optional dependency at runtime
    LoraConfig = None

try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "TRL PPO dependencies are missing. Install requirements_trl.txt first."
    ) from exc


@dataclass
class EpisodeStats:
    episode: int
    reward: float
    trl_updates: int
    correct_detect: int
    false_positive: int
    missed_attack: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_and_tokenizer(args):
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = None
    if args.use_lora:
        if LoraConfig is None:
            raise RuntimeError("peft is not installed but --use_lora was requested.")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        peft_config=peft_config,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    return model, tokenizer


def build_ppo_config(args):
    config_kwargs = {
        "learning_rate": args.learning_rate,
        "batch_size": 1,
        "mini_batch_size": 1,
        "gradient_accumulation_steps": 1,
    }
    if args.use_wandb:
        config_kwargs["log_with"] = "wandb"

    # TRL versions expose slightly different PPOConfig signatures.
    supported = set(inspect.signature(PPOConfig.__init__).parameters.keys())
    filtered = {k: v for k, v in config_kwargs.items() if k in supported}
    missing = set(config_kwargs.keys()) - set(filtered.keys())
    if missing:
        print(f"[TRL] Skipping unsupported PPOConfig args for this TRL version: {sorted(missing)}")
    return PPOConfig(**filtered)


def build_ppo_trainer(ppo_config, model, tokenizer):
    trainer_kwargs = {
        "config": ppo_config,
        "model": model,
        "ref_model": None,
        "tokenizer": tokenizer,
    }
    # TRL versions also differ on PPOTrainer init args.
    supported = set(inspect.signature(PPOTrainer.__init__).parameters.keys())
    filtered = {k: v for k, v in trainer_kwargs.items() if k in supported}
    missing = set(trainer_kwargs.keys()) - set(filtered.keys())
    if missing:
        print(f"[TRL] Skipping unsupported PPOTrainer args for this TRL version: {sorted(missing)}")
    return PPOTrainer(**filtered)


def train(args):
    if args.use_unsloth:
        # Delegate to low-memory Unsloth path for Colab/disk-constrained setups.
        from .train_trl_unsloth import train_unsloth

        print("[TRL] --use_unsloth enabled. Switching to Unsloth training path.")
        train_unsloth(args)
        return

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(args)

    ppo_config = build_ppo_config(args)
    ppo_trainer = build_ppo_trainer(ppo_config, model, tokenizer)

    env = MarketEnv()
    history: List[Dict] = []
    best_reward = -float("inf")

    for episode in range(args.episodes):
        stage = min(5, 1 + episode // max(1, args.stage_span))
        obs = env.reset(stage=stage, seed=args.seed + episode)
        done = False

        ep_reward = 0.0
        ep_updates = 0
        ep_correct = 0
        ep_fp = 0
        ep_missed = 0

        while not done:
            prompt = observation_to_prompt(obs)
            query_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.pretrained_model.device
            )

            with torch.no_grad():
                full_ids = model.pretrained_model.generate(
                    query_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response_ids = full_ids[:, query_ids.shape[1] :]
            response_text = tokenizer.decode(
                response_ids[0], skip_special_tokens=True
            ).strip()
            text_action = parse_model_action(response_text)
            env_action = text_action_to_env_action(text_action, obs)

            next_obs, env_reward, done, info = env.step(env_action)
            trl_reward = to_trl_reward(env_reward, info, text_action)

            ppo_trainer.step(
                [query_ids[0].detach().cpu()],
                [response_ids[0].detach().cpu()],
                [torch.tensor(float(trl_reward), dtype=torch.float32)],
            )

            ep_reward += float(trl_reward)
            ep_updates += 1
            ep_correct += int(info.get("correct_detect", 0))
            ep_fp += int(info.get("false_positive", 0))
            ep_missed += int(info.get("missed_attack", 0))

            obs = next_obs

        stats = EpisodeStats(
            episode=episode,
            reward=ep_reward,
            trl_updates=ep_updates,
            correct_detect=ep_correct,
            false_positive=ep_fp,
            missed_attack=ep_missed,
        )
        history.append(asdict(stats))

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

        if (episode + 1) % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-ep{episode+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

        if (episode + 1) % 5 == 0:
            print(
                f"[TRL] Episode {episode+1}/{args.episodes} "
                f"| Stage {stage} | Reward {ep_reward:.3f} "
                f"| TP {ep_correct} FP {ep_fp} FN {ep_missed}"
            )

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    metrics_path = os.path.join(
        args.output_dir, f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Artifacts saved under: {args.output_dir}")
    print(f"History saved: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TradeX Overseer with TRL PPO.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--stage_span", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="models/trl_overseer")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--use_unsloth",
        action="store_true",
        help="Use the Unsloth low-memory training path (recommended for Colab disk/VRAM limits).",
    )
    parser.add_argument(
        "--bootstrap_episodes",
        type=int,
        default=40,
        help="Used by Unsloth path to generate bootstrap training examples from the environment.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Used by Unsloth path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Used by Unsloth path.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Used by Unsloth path.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Used by Unsloth path.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Used by Unsloth path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

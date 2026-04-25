import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compare import run_evaluation
from .env import MarketEnv
from .text_adapter import observation_to_prompt, parse_model_action, text_action_to_env_action


def load_trl_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def run_trl_evaluation(model, tokenizer, num_episodes=30):
    env = MarketEnv()
    final_prices = []
    rewards = []
    bots_blocked = []
    missed_attacks = []
    false_positives = []
    action_counts = {"ALLOW": 0, "BLOCK_0": 0, "BLOCK_1": 0, "BLOCK_2": 0, "BLOCK_3": 0}

    for ep in range(num_episodes):
        obs = env.reset(stage=5, seed=2000 + ep)
        done = False
        total_reward = 0.0

        tp = 0
        fp = 0
        fn = 0

        while not done:
            prompt = observation_to_prompt(obs)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                full_ids = model.generate(
                    input_ids,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            response_ids = full_ids[:, input_ids.shape[1] :]
            text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
            text_action = parse_model_action(text)
            env_action = text_action_to_env_action(text_action, obs)

            obs, reward, done, info = env.step(env_action)
            total_reward += reward

            action_counts[env_action] = action_counts.get(env_action, 0) + 1
            tp += int(info.get("correct_detect", 0))
            fp += int(info.get("false_positive", 0))
            fn += int(info.get("missed_attack", 0))

        final_prices.append(info["final_price"])
        rewards.append(total_reward)
        bots_blocked.append(tp)
        false_positives.append(fp)
        missed_attacks.append(fn)

    total_actions = sum(action_counts.values())
    intervention_rate = (
        (total_actions - action_counts.get("ALLOW", 0)) / total_actions * 100.0
        if total_actions > 0
        else 0.0
    )
    tp_sum = sum(bots_blocked)
    fp_sum = sum(false_positives)
    fn_sum = sum(missed_attacks)
    precision = (tp_sum / (tp_sum + fp_sum)) * 100.0 if (tp_sum + fp_sum) > 0 else 0.0
    recall = (tp_sum / (tp_sum + fn_sum)) * 100.0 if (tp_sum + fn_sum) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "avg_final_price_error": np.mean([abs(p - 100.0) for p in final_prices]),
        "price_std": np.std(final_prices),
        "avg_reward": np.mean(rewards),
        "bots_blocked": np.mean(bots_blocked),
        "missed_attacks": np.mean(missed_attacks),
        "false_positives": np.mean(false_positives),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "intervention_rate": intervention_rate,
    }


def main(args):
    print("\nEvaluating PPO baseline...")
    ppo_results = run_evaluation(num_episodes=args.episodes, use_overseer=True, deterministic=True)
    print("Evaluating TRL overseer...")
    model, tokenizer = load_trl_model(args.model_path)
    trl_results = run_trl_evaluation(model, tokenizer, num_episodes=args.episodes)

    print("\n" + "=" * 95)
    print("                          PPO vs TRL OVERSEER EVALUATION")
    print("=" * 95)
    print("Metric                      | PPO (Current) | TRL (LLM)")
    print("-" * 95)
    print(f"Avg Reward                  | {ppo_results['avg_reward']:12.2f} | {trl_results['avg_reward']:9.2f}")
    print(f"Price Error                 | {ppo_results['avg_final_price_error']:12.2f} | {trl_results['avg_final_price_error']:9.2f}")
    print(f"Volatility (Std)            | {ppo_results['price_std']:12.2f} | {trl_results['price_std']:9.2f}")
    print(f"Correct Blocks (TP)         | {ppo_results['bots_blocked']:12.2f} | {trl_results['bots_blocked']:9.2f}")
    print(f"Missed Attacks (FN)         | {ppo_results['missed_attacks']:12.2f} | {trl_results['missed_attacks']:9.2f}")
    print(f"False Positives (FP)        | {ppo_results['false_positives']:12.2f} | {trl_results['false_positives']:9.2f}")
    print(f"Intervention Rate (%)       | {ppo_results['intervention_rate']:12.2f} | {trl_results['intervention_rate']:9.2f}")
    print(f"Precision (%)               | {ppo_results['precision']:12.2f} | {trl_results['precision']:9.2f}")
    print(f"Recall (%)                  | {ppo_results['recall']:12.2f} | {trl_results['recall']:9.2f}")
    print(f"F1 Score (%)                | {ppo_results['f1_score']:12.2f} | {trl_results['f1_score']:9.2f}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TRL overseer vs PPO baseline")
    parser.add_argument("--model_path", type=str, default="models/trl_overseer/final")
    parser.add_argument("--episodes", type=int, default=30)
    main(parser.parse_args())

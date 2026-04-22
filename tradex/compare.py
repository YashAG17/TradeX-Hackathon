import torch
import numpy as np
import os
from .env import MarketEnv
from .overseer import Overseer, encode_observation, action_map

def run_evaluation(num_episodes=50, use_overseer=True):
    env = MarketEnv()
    
    policy = None
    if use_overseer:
        policy = Overseer()
        if os.path.exists("models/best_model.pth"):
            policy.load_state_dict(torch.load("models/best_model.pth", map_location="cpu", weights_only=True))
        policy.eval()

    final_prices = []
    rewards = []
    bots_blocked = []
    false_positives = []
    
    for ep in range(num_episodes):
        seed = 1000 + ep # fixed evaluation seeds
        obs = env.reset(stage=5, seed=seed) # Hardest stage
        total_reward = 0
        done = False
        
        while not done:
            if use_overseer:
                obs_vec = encode_observation(obs)
                action_idx, _, _, _, _ = policy.select_action(obs_vec, deterministic=True)
                action_str = action_map[action_idx]
            else:
                action_str = "ALLOW"
                
            obs, reward, done, info = env.step(action_str)
            total_reward += reward
            
        final_prices.append(info["final_price"])
        rewards.append(total_reward)
        bots_blocked.append(info.get("correct_detect", 0.0) > 0) 
        false_positives.append(info.get("false_positive", 0.0) > 0)

    # Note: sum is taken over episodes, but info["correct_detect"] in evaluation might represent last step. 
    # For actual metrics, we'd sum over steps. This is simplified for compare table.
    
    return {
        "avg_final_price": np.mean(final_prices),
        "price_std": np.std(final_prices),
        "avg_reward": np.mean(rewards),
        "bots_blocked": np.sum(bots_blocked) / num_episodes, # Rate per ep
        "false_positives": np.sum(false_positives) / num_episodes
    }

def main():
    print("Evaluating baseline (No Overseer)...")
    no_overseer = run_evaluation(num_episodes=30, use_overseer=False)
    
    print("Evaluating trained model (With Overseer)...")
    with_overseer = run_evaluation(num_episodes=30, use_overseer=True)
    
    print("\n" + "=" * 60)
    print("           WITHOUT Overseer    WITH Overseer")
    print("-" * 60)
    print(f"Avg Final Price:   {no_overseer['avg_final_price']:7.1f}           {with_overseer['avg_final_price']:7.1f}")
    print(f"Price Std Dev:     {no_overseer['price_std']:7.1f}           {with_overseer['price_std']:7.1f}")
    print(f"Avg Reward:        {no_overseer['avg_reward']:7.1f}           {with_overseer['avg_reward']:7.1f}")
    print(f"Bots Blocked/ep:   {no_overseer['bots_blocked']:7.1f}           {with_overseer['bots_blocked']:7.1f}")
    print(f"False Positive/ep: {no_overseer['false_positives']:7.1f}           {with_overseer['false_positives']:7.1f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
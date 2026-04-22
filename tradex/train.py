import argparse
import random
import os
import torch
import torch.optim as optim
import numpy as np
import json

from .env import MarketEnv
from .overseer import Overseer, encode_observation, action_map
from .utils import plot_all_metrics, save_episode_log

def get_stage(episode):
    # Difficulty escalates every 25 episodes
    return min(5, (episode // 25) + 1)

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Normalize
    if returns.std() > 1e-5:
        returns = (returns - returns.mean()) / returns.std()
    return returns

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.onsite else "cpu")
    print(f"Training on device: {device}")
    
    env = MarketEnv()
    policy = Overseer().to(device)
    
    # Checkpoint loading
    start_ep = 0
    if os.path.exists("models/best_model.pth"):
        print("Loading existing checkpoint...")
        policy.load_state_dict(torch.load("models/best_model.pth", map_location=device, weights_only=True))
        
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    metrics_history = []
    best_reward = -float('inf')
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # Training Loop
    for episode in range(start_ep, args.episodes):
        stage = get_stage(episode)
        seed = 42 + episode
        obs = env.reset(stage=stage, seed=seed)
        
        log_probs = []
        values = []
        rewards = []
        entropies = []
        
        ep_false_positives = 0
        ep_correct_blocks = 0
        ep_total_reward = 0.0
        
        step_logs = []
        done = False
        
        # Verbose logging for specific episodes
        verbose = args.verbose or (episode % 50 == 0)
        if verbose:
            print(f"\nEpisode {episode} | Seed {seed} | Stage {stage}")
            print("-" * 50)
            
        while not done:
            obs_vec = encode_observation(obs)
            action_idx, log_prob, val, entropy, probs_np = policy.select_action(obs_vec)
            action_str = action_map[action_idx]
            
            obs, reward, done, info = env.step(action_str)
            
            ep_false_positives += info["false_positive"]
            ep_correct_blocks += info["correct_detect"]
            ep_total_reward += reward
            
            log_probs.append(log_prob)
            values.append(val)
            rewards.append(reward)
            entropies.append(entropy)
            
            if verbose:
                print(f"Step {env.timestep:02d}")
                print(f"Price: {env.price:.1f} | Liquidity: {info.get('total_liquidity',0):.0f}")
                print(f"Reward: {reward:.2f} | CumReward: {ep_total_reward:.2f}")
                print("Agents:")
                for trader in info["step_trades"]:
                    print(f"A{trader['agent']} {info['agent_types'][trader['agent']]:18} {trader['action']:4} {trader['size']:5.1f}")
                print(f"Overseer Action = {action_str} | Probs = {[round(p,2) for p in probs_np]}")
                if info["is_block"]:
                    if info["correct_detect"] > 0:
                        print("Result: Correct block inferred from behavior\n")
                    else:
                        print("Result: FALSE POSITIVE\n")
                else:
                    print("\n")
            
            step_logs.append({
                "timestep": env.timestep,
                "price": float(env.price),
                "action": action_str,
                "reward": float(reward),
                "stats": [dict(s) for s in obs["stats"]]
            })
            
        returns = compute_returns(rewards)
        
        # PPO / A2C inspired update
        policy_loss = 0
        value_loss = 0
        entropy_bonus = 0
        
        for log_prob, val, R, ent in zip(log_probs, values, returns, entropies):
            advantage = R - val.item()
            policy_loss -= log_prob * advantage
            value_loss += 0.5 * (val.squeeze(-1) - R).pow(2)
            entropy_bonus -= 0.01 * ent
            
        loss = policy_loss + value_loss + entropy_bonus
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
        
        ep_metrics = {
            "episode": episode,
            "seed": seed,
            "stage": stage,
            "reward": float(ep_total_reward),
            "false_positives": ep_false_positives,
            "bots_blocked": ep_correct_blocks,
            "price_error": info["price_error"]
        }
        metrics_history.append(ep_metrics)
        
        if verbose:
            print("-" * 50)
            
        if (episode + 1) % 10 == 0:
            avg_rew = np.mean([m['reward'] for m in metrics_history[-10:]])
            print(f"Ep {episode+1:4d} | Score: {avg_rew:5.2f} | Stage: {stage} | FP: {ep_false_positives:.0f} | Blocked: {ep_correct_blocks:.0f}")
            if avg_rew > best_reward:
                best_reward = avg_rew
                torch.save(policy.state_dict(), "models/best_model.pth")
                
        # Export episode log
        save_episode_log(episode, seed, {
            "metrics": ep_metrics,
            "steps": step_logs
        })
        
    # Save final models and metrics
    with open("metrics/training_history.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
        
    plot_all_metrics(metrics_history)
    print("\nTraining Complete! Plots saved to /plots. Best model in /models.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--onsite", action="store_true", help="Run with GPU and extended episodes for onsite event")
    parser.add_argument("--verbose", action="store_true", help="Print all steps")
    args = parser.parse_args()
    
    if args.onsite:
        args.episodes = max(args.episodes, 1000)
        
    train(args)
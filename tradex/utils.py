import matplotlib.pyplot as plt
import numpy as np
import os
import json

def plot_all_metrics(metrics_history, export_dir='plots'):
    os.makedirs(export_dir, exist_ok=True)
    
    episodes = [m['episode'] for m in metrics_history]
    rewards = [m['reward'] for m in metrics_history]
    fps = [m['false_positives'] for m in metrics_history]
    blocked = [m['bots_blocked'] for m in metrics_history]
    errors = [m['price_error'] for m in metrics_history]
    
    # 1. Reward
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label='Episode Reward', color="#1f77b4")
    if len(rewards) >= 10:
        plt.plot(episodes[9:], np.convolve(rewards, np.ones(10)/10, mode='valid'), color="#ff7f0e", label='Rolling Avg')
    plt.title('Reward vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{export_dir}/reward_vs_episode.png')
    plt.close()
    
    # 2. False Positives
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, fps, color="#d62728")
    plt.title('False Positives vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('False Positives Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{export_dir}/false_positives_vs_episode.png')
    plt.close()
    
    # 3. Bots Blocked
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, blocked, color="#2ca02c")
    plt.title('Correct Blocks vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Blocks Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{export_dir}/bots_blocked_vs_episode.png')
    plt.close()
    
    # 4. Price Error
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, errors, color="#9467bd")
    plt.title('Final Price Error vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Abs Deviation from Baseline')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{export_dir}/final_price_error_vs_episode.png')
    plt.close()

def save_episode_log(episode, seed, log_data):
    os.makedirs('logs', exist_ok=True)
    with open(f"logs/episode_{episode}_seed_{seed}.json", "w") as f:
        json.dump(log_data, f, indent=2)
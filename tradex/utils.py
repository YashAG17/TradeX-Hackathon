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
    
    # Precision/Recall (Rolling calculation)
    precisions = []
    recalls = []
    
    for i in range(len(episodes)):
        tp = blocked[i]
        fp = fps[i]
        
        # Estimate: True attacks vary, assume 4 per stage
        fn = max(0, 4 - tp)
        
        if (tp + fp) > 0:
            precisions.append((tp / (tp + fp)) * 100)
        else:
            precisions.append(0.0 if len(precisions)==0 else precisions[-1])
            
        if (tp + fn) > 0:
            recalls.append((tp / (tp + fn)) * 100)
        else:
            recalls.append(0.0 if len(recalls)==0 else recalls[-1])
    
    # 1. Reward
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label='Episode Reward', color="#1f77b4", alpha=0.3)
    if len(rewards) >= 20:
        plt.plot(episodes[19:], np.convolve(rewards, np.ones(20)/20, mode='valid'), color="#17becf", label='Rolling Avg', linewidth=2)
    plt.title('Reward vs Episodes (RL Objective)')
    plt.xlabel('Episodes')
    plt.ylabel('Total Episode Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{export_dir}/reward_vs_episode.png', dpi=150)
    plt.close()
    
    # 2. Precision & Recall Tracking
    plt.figure(figsize=(10, 5))
    if len(precisions) >= 20:
        plt.plot(episodes[19:], np.convolve(precisions, np.ones(20)/20, mode='valid'), color="#2ca02c", label='Precision %', linewidth=2)
        plt.plot(episodes[19:], np.convolve(recalls, np.ones(20)/20, mode='valid'), color="#ff7f0e", label='Recall %', linewidth=2)
    else:
        plt.plot(episodes, precisions, color="#2ca02c", label='Precision %')
        plt.plot(episodes, recalls, color="#ff7f0e", label='Recall %')
    plt.title('Precision vs Recall Tracking')
    plt.xlabel('Episodes')
    plt.ylabel('Percentage %')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{export_dir}/precision_recall.png', dpi=150)
    plt.close()
    
    # 3. Bots Blocked / False Positives Area
    plt.figure(figsize=(10, 5))
    if len(blocked) >= 20:
        smoothed_fp = np.convolve(fps, np.ones(20)/20, mode='valid')
        smoothed_tp = np.convolve(blocked, np.ones(20)/20, mode='valid')
        plt.fill_between(episodes[19:], 0, smoothed_tp, color="#98df8a", label="Correct Blocks (TP)", alpha=0.6)
        plt.plot(episodes[19:], smoothed_fp, color="#d62728", label="False Positives (FP)", linewidth=2.5)
    plt.title('Detection Capability (Smoothed Area)')
    plt.xlabel('Episodes')
    plt.ylabel('Event Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{export_dir}/detection_capability.png', dpi=150)
    plt.close()
    
    # 4. Price Error
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, errors, color="#9467bd", alpha=0.3)
    if len(errors) >= 20:
        plt.plot(episodes[19:], np.convolve(errors, np.ones(20)/20, mode='valid'), color="#9467bd", label='Rolling Avg', linewidth=2)
    plt.title('Final Price Error vs Episode (Market Stability)')
    plt.xlabel('Episodes')
    plt.ylabel('Abs Deviation from Baseline Target')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{export_dir}/final_price_error_vs_episode.png', dpi=150)
    plt.close()

def save_episode_log(episode, seed, log_data):
    os.makedirs('logs', exist_ok=True)
    with open(f"logs/episode_{episode}_seed_{seed}.json", "w") as f:
        json.dump(log_data, f, indent=2)
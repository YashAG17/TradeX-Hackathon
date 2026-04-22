import gradio as gr
import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tradex.env import MarketEnv
from tradex.overseer import Overseer, encode_observation, action_map

# Load best policy globally if exists
policy = Overseer()
if os.path.exists("models/best_model.pth"):
    policy.load_state_dict(torch.load("models/best_model.pth", map_location="cpu", weights_only=True))
policy.eval()

def run_single_episode(seed, stage, use_overseer):
    env = MarketEnv()
    obs = env.reset(stage=int(stage), seed=int(seed))
    
    logs = []
    total_reward = 0
    
    done = False
    while not done:
        if use_overseer:
            obs_vec = encode_observation(obs)
            action_idx, _, _, _, probs = policy.select_action(obs_vec, deterministic=True)
            action_str = action_map[action_idx]
        else:
            action_str = "ALLOW"
            
        obs, reward, done, info = env.step(action_str)
        total_reward += reward
        
        step_log = f"Step {env.timestep:02d} | Price: {env.price:.1f} | Action: {action_str} | Reward: {reward:.2f}\n"
        step_log += "Agents:\n"
        for t in info["step_trades"]:
            step_log += f"  A{t['agent']} ({info['agent_types'][t['agent']]}) -> {t['action']} {t['size']:.1f}\n"
        logs.append(step_log)
        
    summary = f"""### Episode Complete
**Final Price:** {env.price:.2f}
**Total Reward:** {total_reward:.2f}
**Stage:** {stage}
"""
    return summary, "\n".join(logs)

def load_plot(plot_name):
    path = f"plots/{plot_name}"
    if os.path.exists(path):
        return path
    return None

with gr.Blocks() as demo:
    gr.Markdown("# 📈 TradeX — AI Governance for Autonomous Markets")
    gr.Markdown("A multi-agent autonomous market simulation controlled by an AI Overseer.")
    
    with gr.Tab("Market Replay"):
        with gr.Row():
            with gr.Column(scale=1):
                seed_input = gr.Number(value=42, label="Random Seed")
                stage_input = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Difficulty Stage")
                use_ai = gr.Checkbox(value=True, label="Enable AI Overseer")
                run_btn = gr.Button("Run Episode", variant="primary")
            
            with gr.Column(scale=2):
                ep_summary = gr.Markdown("Run an episode to see results here.")
                ep_logs = gr.Textbox(label="Step-by-Step Logs", lines=15, max_lines=20)
                
        run_btn.click(run_single_episode, inputs=[seed_input, stage_input, use_ai], outputs=[ep_summary, ep_logs])
        
    with gr.Tab("Training Metrics & Plots"):
        gr.Markdown("View the performance curves during RL training.")
        
        with gr.Row():
            p1 = gr.Image(value=load_plot("reward_vs_episode.png"), label="Reward Curve")
            p2 = gr.Image(value=load_plot("false_positives_vs_episode.png"), label="False Positives")
            
        with gr.Row():
            p3 = gr.Image(value=load_plot("bots_blocked_vs_episode.png"), label="Bots Blocked")
            p4 = gr.Image(value=load_plot("final_price_error_vs_episode.png"), label="Price Stability")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Monochrome(primary_hue="blue", secondary_hue="slate"))

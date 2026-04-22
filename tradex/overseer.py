import torch
import torch.nn as nn
import numpy as np

class Overseer(nn.Module):
    def __init__(self, obs_dim=36, hidden_dim=128, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        h = self.net(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value
        
    def select_action(self, obs_vec, deterministic=False):
        x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        logits, val = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
            return action.item(), None, val, None, probs[0].detach().numpy()
            
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), val, dist.entropy(), probs[0].detach().numpy()

def encode_observation(obs):
    # Vector length: 
    # price (1), hist (10), timestep/50 (1)
    # stats per agent: buys/10, sells/10, vol/50, blocked/10, burst, impact (6 * 4 = 24)
    # total: 1+10+1+24 = 36
    v = [obs['price']/100.0]
    v += [p/100.0 for p in obs['price_history']]
    v += [obs['timestep']/50.0]
    for i in range(4):
        s = obs['stats'][i]
        v += [
            s['buys']/10.0, 
            s['sells']/10.0, 
            s['vol']/50.0, 
            s['blocked']/10.0,
            s['burst_score']/20.0,
            s['impact_ratio']
        ]
    return np.array(v, dtype=np.float32)

action_map = ["ALLOW", "BLOCK_0", "BLOCK_1", "BLOCK_2", "BLOCK_3"]
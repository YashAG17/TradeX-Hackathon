import numpy as np
from .agents import NormalTrader, ManipulatorBot, ArbitrageAgent, LiquidityProvider
from .reward import compute_reward

class MarketEnv:
    def __init__(self):
        self.max_steps = 50
        self.baseline_price = 100.0

    def reset(self, stage=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.stage = stage
        self.timestep = 0
        
        # Constant Product AMM: x * y = k
        self.reserve_y = 100000.0
        self.price = self.baseline_price
        self.reserve_x = self.reserve_y / self.price
        self.k = self.reserve_x * self.reserve_y
        
        # Randomize agent identities but ensure all roles present
        types = [NormalTrader, ManipulatorBot, ArbitrageAgent, LiquidityProvider]
        np.random.shuffle(types)
        self.agents = [t(i) for i, t in enumerate(types)]
        
        self.price_history = [self.baseline_price] * 10
        self.trade_history = []
        
        # Behavior stats array representing suspicion features without exposing logic
        self.agent_stats = [{"buys": 0, "sells": 0, "vol": 0, "blocked": 0, "burst_score": 0.0, "impact_ratio": 0.0} for _ in range(4)]
        self.last_action_blocked = [False] * 4
        
        return self._get_obs()

    def step(self, overseer_action):
        self.timestep += 1
        
        # Action map: "ALLOW", "BLOCK_0", "BLOCK_1", "BLOCK_2", "BLOCK_3"
        blocked_agent = -1
        if overseer_action.startswith("BLOCK_"):
            try:
                blocked_agent = int(overseer_action.split("_")[1])
            except:
                pass
                
        step_trades = []
        true_malicious_actions = [0.0] * 4
        
        for i, agent in enumerate(self.agents):
            if i == blocked_agent:
                self.last_action_blocked[i] = True
                self.agent_stats[i]["blocked"] += 1
                continue
                
            self.last_action_blocked[i] = False
            action, size = agent.act(self.price, self.timestep, self.stage, self.last_action_blocked[i])
            if size > 0:
                step_trades.append({"agent": i, "action": action, "size": size})
                if agent.is_malicious:
                    true_malicious_actions[i] += size
        
        # Execute AMM
        total_buy = sum(t["size"] for t in step_trades if t["action"] == "BUY")
        total_sell = sum(t["size"] for t in step_trades if t["action"] == "SELL")
        
        old_price = self.price
        # Apply trades
        if total_buy > 0:
            dy = total_buy * self.price
            self.reserve_y += dy
            self.reserve_x = self.k / self.reserve_y
        if total_sell > 0:
            dx = total_sell
            self.reserve_x += dx
            self.reserve_y = self.k / self.reserve_x
            
        self.price = self.reserve_y / self.reserve_x
        self.price_history.append(self.price)
        self.price_history.pop(0)
        
        price_impact = abs(self.price - old_price)
        
        # Update behavioral metrics (Suspicion Engine mechanics)
        for t in step_trades:
            idx = t["agent"]
            if t["action"] == "BUY":
                self.agent_stats[idx]["buys"] += 1
            else:
                self.agent_stats[idx]["sells"] += 1
            self.agent_stats[idx]["vol"] += t["size"]
            
            # Simple burst/impact heuristics for observation
            self.agent_stats[idx]["burst_score"] = self.agent_stats[idx]["burst_score"] * 0.8 + 0.2 * t["size"]
            self.agent_stats[idx]["impact_ratio"] = self.agent_stats[idx]["impact_ratio"] * 0.9 + 0.1 * price_impact
            
        self.trade_history.append(step_trades)
        done = self.timestep >= self.max_steps
        
        malicious_ids = [i for i, a in enumerate(self.agents) if getattr(a, 'is_malicious', False)]
        
        reward, info = compute_reward(
            overseer_action=overseer_action,
            malicious_ids=malicious_ids,
            price=self.price,
            baseline_price=self.baseline_price,
            blocked_agent=blocked_agent,
            true_malicious_actions=true_malicious_actions
        )
        
        info["step_trades"] = step_trades
        info["final_price"] = self.price
        info["total_liquidity"] = self.reserve_x + self.reserve_y
        info["agent_types"] = {i: a.__class__.__name__ for i, a in enumerate(self.agents)} # For logs only
        
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        # Global state + per-agent stats
        return {
            "price": self.price,
            "price_history": list(self.price_history),
            "timestep": self.timestep,
            "stage": self.stage,
            "stats": self.agent_stats
        }
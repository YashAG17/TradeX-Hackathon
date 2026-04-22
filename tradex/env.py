import numpy as np
import random
from .agents import NormalTrader, NoisyTrader, Manipulator, Arbitrage
from .reward import compute_reward

class MarketEnv:
    def __init__(self):
        self.max_steps = 50
        self.baseline_price = 100.0

    def reset(self, stage=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.stage = stage
        self.timestep = 0
        
        self.reserve_y = 100000.0
        self.price = self.baseline_price
        self.reserve_x = self.reserve_y / self.price
        self.k = self.reserve_x * self.reserve_y
        
        # Guarantee 1 Manipulator, pick 3 benign randomly
        benign_pool = [NormalTrader, NoisyTrader, Arbitrage]
        benign_selection = random.choices(benign_pool, k=3)
        types = [Manipulator] + benign_selection
        
        # VERY IMPORTANT: Shuffle identities perfectly so index != role
        np.random.shuffle(types)
        
        self.agents = [t(i) for i, t in enumerate(types)]
        
        self.price_history = [self.baseline_price] * 10
        self.trade_history = []
        
        self.agent_stats = [{"buys": 0, "sells": 0, "avg_size": 0.0, "blocked": 0, 
                             "burst_score": 0.0, "pump_score": 0.0, "dump_score": 0.0,
                             "coordination": 0.0, "pnl": 0.0} for _ in range(4)]
        self.last_action_blocked = [False] * 4
        
        return self._get_obs()

    def step(self, overseer_action):
        self.timestep += 1
        
        blocked_agent = -1
        if overseer_action.startswith("BLOCK_"):
            try:
                blocked_agent = int(overseer_action.split("_")[1])
            except:
                pass
                
        step_trades = []
        true_malicious_actions = [0.0] * 4
        block_reason = None
        
        for i, agent in enumerate(self.agents):
            self.last_action_blocked[i] = False
            # Generate the intended action first to see what they wanted to do
            action, size = agent.act(self.price, self.timestep, self.stage, self.last_action_blocked[i], self.price_history)
            
            # Record malicious intent if they attempted to execute a trade
            if agent.is_malicious and size > 0:
                true_malicious_actions[i] += size
                
            # Now intercept and cancel the trade if they were blocked
            if i == blocked_agent:
                self.last_action_blocked[i] = True
                self.agent_stats[i]["blocked"] += 1
                
                # Determine WHY they were blocked for the judges
                stats = self.agent_stats[i]
                if stats["burst_score"] > 8.0:
                    block_reason = "repeated burst buys"
                elif stats["dump_score"] > 20.0:
                    block_reason = "abnormal price impact"
                elif stats["pump_score"] > 30.0:
                    block_reason = "pump then dump sequence"
                elif abs(self.price - self.baseline_price) > 5.0:
                    block_reason = "liquidity drain / disruption"
                else:
                    block_reason = "spoof-like order pattern"
                    
                continue # Trade is TRULY CANCELLED, does not append to step_trades
                
            if size > 0:
                step_trades.append({"agent": i, "action": action, "size": size})
        
        total_buy = sum(t["size"] for t in step_trades if t["action"] == "BUY")
        total_sell = sum(t["size"] for t in step_trades if t["action"] == "SELL")
        
        old_price = self.price
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
        
        for t in step_trades:
            idx = t["agent"]
            sz = t["size"]
            total_trades = self.agent_stats[idx]["buys"] + self.agent_stats[idx]["sells"]
            
            self.agent_stats[idx]["avg_size"] = (self.agent_stats[idx]["avg_size"] * total_trades + sz) / (total_trades + 1)
            
            if t["action"] == "BUY":
                self.agent_stats[idx]["buys"] += 1
                self.agent_stats[idx]["pump_score"] += sz
                self.agent_stats[idx]["pnl"] -= sz * self.price
            else:
                self.agent_stats[idx]["sells"] += 1
                self.agent_stats[idx]["dump_score"] += sz
                self.agent_stats[idx]["pnl"] += sz * self.price
                
            self.agent_stats[idx]["burst_score"] = self.agent_stats[idx]["burst_score"] * 0.5 + sz
            
        self.trade_history.append(step_trades)
        done = self.timestep >= self.max_steps
        
        malicious_ids = [i for i, a in enumerate(self.agents) if getattr(a, 'is_malicious', False)]
        
        reward, info = compute_reward(
            overseer_action=overseer_action,
            malicious_ids=malicious_ids,
            agents=self.agents,
            price=self.price,
            baseline_price=self.baseline_price,
            blocked_agent=blocked_agent,
            true_malicious_actions=true_malicious_actions
        )
        
        info["step_trades"] = step_trades
        info["final_price"] = self.price
        info["total_liquidity"] = self.reserve_x + self.reserve_y
        info["agent_types"] = {i: a.__class__.__name__ for i, a in enumerate(self.agents)}
        info["agent_strategies"] = {i: getattr(a, 'strategy', 'N/A') for i, a in enumerate(self.agents)}
        info["block_reason"] = block_reason
        
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        prices = np.array(self.price_history)
        momentum = prices[-1] - prices[-2]
        volatility = np.std(prices)
        price_dev = abs(self.price - self.baseline_price)
        
        return {
            "price": self.price,
            "price_history": list(self.price_history),
            "momentum": momentum,
            "volatility": volatility,
            "liquidity": self.reserve_x + self.reserve_y,
            "price_dev": price_dev,
            "timestep": self.timestep,
            "stage": self.stage,
            "stats": self.agent_stats
        }
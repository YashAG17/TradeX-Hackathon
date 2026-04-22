import numpy as np

class Agent:
    def __init__(self, obj_id):
        self.obj_id = obj_id
        self.is_malicious = False

    def act(self, price, timestep, stage, last_action_blocked):
        return "HOLD", 0.0

class NormalTrader(Agent):
    def act(self, price, timestep, stage, last_action_blocked):
        if np.random.rand() < 0.2:
            return "BUY", np.random.uniform(1.0, 5.0)
        elif np.random.rand() < 0.2:
            return "SELL", np.random.uniform(1.0, 5.0)
        return "HOLD", 0.0

class ManipulatorBot(Agent):
    def __init__(self, obj_id):
        super().__init__(obj_id)
        self.is_malicious = True

    def act(self, price, timestep, stage, last_action_blocked):
        if stage == 1:
            # Obvious attacks early
            if 10 <= timestep <= 15:
                return "BUY", np.random.uniform(15.0, 25.0)
            if timestep == 25:
                return "SELL", 80.0
        elif stage == 2:
            # Stealth later
            if 10 <= timestep <= 20 and timestep % 2 == 0:
                return "BUY", np.random.uniform(2.0, 4.0)
            if 30 <= timestep <= 35:
                return "SELL", np.random.uniform(5.0, 10.0)
        elif stage >= 3:
            # Adapts if blocked often + colludes conceptually
            if last_action_blocked:
                return "HOLD", 0.0
            if 5 <= timestep <= 40 and timestep % 4 == 0:
                return "BUY", np.random.uniform(3.0, 6.0)
            if timestep > 42:
                return "SELL", 30.0
        return "HOLD", 0.0

class ArbitrageAgent(Agent):
    def act(self, price, timestep, stage, last_action_blocked):
        target = 100.0
        if price < target * 0.98:
            return "BUY", min(25.0, (target - price) * 5)
        elif price > target * 1.02:
            return "SELL", min(25.0, (price - target) * 5)
        return "HOLD", 0.0

class LiquidityProvider(Agent):
    def act(self, price, timestep, stage, last_action_blocked):
        return "HOLD", 0.0
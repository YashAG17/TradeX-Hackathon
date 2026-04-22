def compute_reward(overseer_action, malicious_ids, agents, price, baseline_price, blocked_agent, true_malicious_actions=None):
    raw = 0.5
    
    is_block = blocked_agent != -1
    malicious_active = true_malicious_actions and sum(true_malicious_actions) > 0
    is_malicious_block = is_block and (blocked_agent in malicious_ids) and malicious_active
    
    correct_detect = 0
    false_positive = 0
    missed_attack = 0
    
    # Core Logic
    if is_block:
        if is_malicious_block:
            raw += 0.45 # Correctly blocking Manipulator
            correct_detect = 1
        else:
            false_positive = 1
            if blocked_agent != -1 and agents:
                agent_name = agents[blocked_agent].__class__.__name__
                if agent_name == "NormalTrader":
                    raw -= 0.25 # Wrongly blocking NormalTrader
                elif agent_name == "Arbitrage":
                    raw -= 0.20 # Wrongly blocking Arbitrage
                else:
                    raw -= 0.25
            else:
                raw -= 0.25
    else: # ALLOW
        if malicious_active:
            raw -= 0.40 # Missing Manipulator attack
            missed_attack = 1
        else:
            raw += 0.12 # Allowing benign NormalTrader / Arbitrage
            
    # Price Stability
    price_error = abs(price - baseline_price) / baseline_price
    if price_error > 0.05:
        raw -= 0.20 # Large final price deviation
        
    # We will simulate high volatility penalization via price_error on a step basis
    if price_error > 0.03:
        raw -= 0.10 # High volatility proxy
        
    if not malicious_active and price_error < 0.01 and not false_positive and not missed_attack:
        raw += 0.10 # Perfect stable episode step
        
    reward = max(0.0, min(1.0, raw))
    
    info = {
        "correct_detect": correct_detect,
        "false_positive": false_positive,
        "missed_attack": missed_attack,
        "price_error": price_error,
        "malicious_ids": malicious_ids,
        "is_block": is_block,
        "malicious_active": malicious_active
    }
    
    return reward, info
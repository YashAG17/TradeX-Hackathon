def compute_reward(overseer_action, malicious_ids, agents, price, baseline_price, blocked_agent, true_malicious_actions=None, threat_score=0.0):
    raw = 0.0
    
    # Only flag as an active attack if the manipulator is throwing significant sizing
    is_attack_active = true_malicious_actions and sum(true_malicious_actions) >= 15.0
    attacker_role = "Manipulator" if is_attack_active else "None"
    
    is_block = blocked_agent != -1
    
    correct_detect = 0
    false_positive = 0
    missed_attack = 0
    wrong_target = 0
    
    if is_attack_active:
        if is_block:
            if blocked_agent in malicious_ids:
                raw += 3.0 # TRUE POSITIVE
                correct_detect = 1
            else:
                raw -= 1.5 # WRONG TARGET
                wrong_target = 1
        else: # ALLOW
            raw -= 3.0 # FALSE NEGATIVE
            missed_attack = 1
    else: # NO ATTACK
        if is_block:
            raw -= 0.7 # FALSE POSITIVE
            false_positive = 1
        else: # ALLOW
            raw += 0.05 # TRUE NEGATIVE
            
    info = {
        "correct_detect": correct_detect,
        "false_positive": false_positive,
        "missed_attack": missed_attack,
        "wrong_target": wrong_target,
        "price_error": abs(price - baseline_price) / baseline_price,
        "malicious_ids": malicious_ids,
        "is_block": is_block,
        "is_attack_active": is_attack_active,
        "attacker_role": attacker_role
    }
    
    return raw, info
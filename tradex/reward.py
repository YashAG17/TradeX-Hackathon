def compute_reward(overseer_action, malicious_ids, price, baseline_price, blocked_agent, true_malicious_actions=None):
    # Normalized reward between 0.0 and 1.0 using weighted shaping
    # reward = 0.35 * correct + 0.20 * allow_normal + 0.20 * stability + 0.10 * low_fp + 0.10 * low_vol + 0.05 * bonus

    correct_detect_score = 0.0
    allow_normal_score = 0.0
    false_positive = 0.0
    
    is_block = blocked_agent != -1
    is_malicious_block = is_block and (blocked_agent in malicious_ids)

    # 1. Correct detection vs false positive
    if is_block:
        if is_malicious_block:
            correct_detect_score = 1.0
        else:
            false_positive = 1.0
            
    # 2. Allowing normal behavior
    # If it was an ALLOW action (no block)
    if not is_block:
        # If there was a malicious attack happening that we allowed, allow_normal_score is low, else 1.0
        if true_malicious_actions and sum(true_malicious_actions) > 0:
            allow_normal_score = 0.0 # missed attack
        else:
            allow_normal_score = 1.0
            
    # 3. Price stability
    price_error = abs(price - baseline_price) / baseline_price
    price_stability_score = max(0.0, 1.0 - (price_error * 10)) # Needs to be tight

    # 4. Low false positive score
    low_fp_score = 1.0 - false_positive
    
    # 5. Low volatility score (simplified to map to price stability for step reward)
    low_volatility_score = price_stability_score
    
    # 6. Episode bonus (computed later or 0.0 per step)
    episode_bonus = 0.0
    
    reward = (
        0.35 * correct_detect_score +
        0.20 * allow_normal_score +
        0.20 * price_stability_score +
        0.10 * low_fp_score +
        0.10 * low_volatility_score +
        0.05 * episode_bonus
    )
    
    # Clip between 0.0 and 1.0
    reward = max(0.0, min(1.0, reward))
    
    info = {
        "correct_detect": correct_detect_score,
        "false_positive": false_positive,
        "price_error": price_error,
        "malicious_ids": malicious_ids,
        "is_block": is_block
    }
    
    return reward, info
from collections import Counter
from typing import List
import math
import numpy as np
from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from collections import defaultdict


def compute_temporal_metrics(solutions: List[str], model_answers: List[str]):
    """
    Compute temporal-specific metrics for tracking training without ground truth.
    
    Args:
        solutions: Raw model output strings
        model_answers: Extracted answers (parsed temporal intervals as strings)
    
    Returns:
        Dict of temporal metrics
    """
    from verl.utils.reward_score.ttrl.qwen.qwen_eval import parse_temporal
    
    # Parse all intervals
    parsed_intervals = []
    valid_count = 0
    
    for ans in model_answers:
        interval = parse_temporal(ans)
        if interval is not None:
            valid_count += 1
            parsed_intervals.append(interval)
    
    metrics = {}
    
    # Parse success rate: % of responses that are valid [start, end] format
    metrics["temporal_parse_rate"] = valid_count / len(model_answers) if model_answers else 0.0
    
    if parsed_intervals:
        starts = [interval[0] for interval in parsed_intervals]
        ends = [interval[1] for interval in parsed_intervals]
        durations = [end - start for start, end in parsed_intervals]
        
        # Average duration: Are predictions reasonable lengths?
        metrics["temporal_avg_duration"] = float(np.mean(durations))
        metrics["temporal_std_duration"] = float(np.std(durations))
        
        # Start/end point statistics
        metrics["temporal_avg_start"] = float(np.mean(starts))
        metrics["temporal_std_start"] = float(np.std(starts))
        metrics["temporal_avg_end"] = float(np.mean(ends))
        metrics["temporal_std_end"] = float(np.std(ends))
        
        # Temporal consistency: lower std = more consistent predictions
        metrics["temporal_consistency"] = 1.0 / (1.0 + float(np.std(starts)) + float(np.std(ends)))
        
        # Invalid interval rate (start > end)
        invalid_order = sum(1 for s, e in parsed_intervals if s > e)
        metrics["temporal_invalid_order_rate"] = invalid_order / len(parsed_intervals)
    else:
        # No valid intervals parsed
        metrics["temporal_avg_duration"] = 0.0
        metrics["temporal_std_duration"] = 0.0
        metrics["temporal_avg_start"] = 0.0
        metrics["temporal_std_start"] = 0.0
        metrics["temporal_avg_end"] = 0.0
        metrics["temporal_std_end"] = 0.0
        metrics["temporal_consistency"] = 0.0
        metrics["temporal_invalid_order_rate"] = 0.0
    
    return metrics


def test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    task="math", extra_info=None):
    
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"

    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)
    counter = Counter(model_answers)
    total = len(model_answers)
    reward_p = [counter[ans] / total for ans in model_answers]


    entropy = 0.0
    for count in counter.values():
        probability = count / total
        if probability > 0:  # Avoid log(0)
            entropy -= probability * math.log(probability)
    
    if total > 1:
        max_entropy = math.log(len(counter))  # Max entropy for this many unique answers
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        normalized_entropy = 0.0
    
    estimated_label, majority_count = counter.most_common(1)[0]
    
    
    hit_rate = 1.0 if auto_verify(task, [estimated_label], [ground_truth], extra_info=extra_info)[0][0] else 0.0
    majority_ratio = majority_count / len(solutions)
    

    rewards, _ = auto_verify(task, solutions, [estimated_label] * len(solutions), extra_info=extra_info)
    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    rewards_en = [(r*1) - (0.75 * normalized_entropy) for r in reward_p]
    
    rewards_hit_rate = 0
    for reward, true_reward in zip(rewards, true_rewards):
        if reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(rewards)

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

    ttrl_metrics = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_ratio": majority_ratio,
        "ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        "majority_voting_reward": sum(rewards) / len(rewards),
        f"pass@{len(solutions)}": 1.0 if sum(true_rewards) >= 1 else 0.0,
        "normalized_entropy": normalized_entropy,
    }
    
    # Add temporal-specific metrics for temporal grounding task
    if task == "tag":
        temporal_metrics = compute_temporal_metrics(solutions, model_answers)
        ttrl_metrics.update(temporal_metrics)
    
    return rewards_en, ttrl_metrics

def post_test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    pred_rewards: List,
    task="math", extra_info=None):
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(solutions) == len(pred_rewards), f"{len(solutions)} vs {len(pred_rewards)}"
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)

    # counter = Counter(model_answers)
    
    # true_label_ratio = counter.get(ground_truth, 0) / len(solutions)

    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)

    # Compare pred_rewards with true_rewards to calculate reward hit rate
    rewards_hit_rate = sum(
        1 if pred == true else 0 for pred, true in zip(pred_rewards, true_rewards)
    ) / len(pred_rewards)



    post_ttrl_metrics = {
        "post_reward_accuracy": rewards_hit_rate,
        "post_ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        f"post_pass@{len(solutions)}": 1.0 if sum(true_rewards) > 0 else 0.0,
    }
    return post_ttrl_metrics
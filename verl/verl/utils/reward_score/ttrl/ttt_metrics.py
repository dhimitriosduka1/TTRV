from collections import Counter
from typing import List
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.qwen.qwen_eval import (
    parse_temporal,
    reward_temporal_iou,
)


def compute_temporal_iou_distance_matrix(intervals):
    """
    Compute pairwise temporal IoU distance matrix.
    Distance = 1 - IoU, so similar segments have low distance.

    Args:
        intervals: List of (start, end) tuples

    Returns:
        np.ndarray: Symmetric distance matrix of shape (n, n)
    """
    n = len(intervals)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            iou = reward_temporal_iou(intervals[i], intervals[j])
            distance = 1.0 - iou
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance

    return dist_matrix


def cluster_temporal_segments(model_answers, distance_threshold=0.10):
    """
    Cluster temporal segments based on IoU similarity using Agglomerative Clustering.

    Args:
        model_answers: List of temporal interval strings like "(start, end)"
        distance_threshold: Maximum distance for merging clusters (1 - min_iou).
                           Default 0.3 means segments with IoU >= 0.7 will be merged.

    Returns:
        tuple: (cluster_labels, parsed_intervals, valid_indices)
            - cluster_labels: Array of cluster labels for valid intervals
            - parsed_intervals: List of parsed (start, end) tuples
            - valid_indices: List of original indices that had valid intervals
    """
    # Parse all intervals
    parsed_intervals = []
    valid_indices = []

    for idx, ans in enumerate(model_answers):
        interval = parse_temporal(ans)
        if interval is not None and interval[0] <= interval[1]:
            parsed_intervals.append(interval)
            valid_indices.append(idx)

    if len(parsed_intervals) == 0:
        return np.array([]), [], []

    if len(parsed_intervals) == 1:
        # Only one valid interval, it's its own cluster
        return np.array([0]), parsed_intervals, valid_indices

    # Compute IoU distance matrix
    dist_matrix = compute_temporal_iou_distance_matrix(parsed_intervals)

    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    ).fit(dist_matrix)

    return clustering.labels_, parsed_intervals, valid_indices


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
    metrics["temporal_parse_rate"] = (
        valid_count / len(model_answers) if model_answers else 0.0
    )

    if parsed_intervals:
        starts = [interval[0] for interval in parsed_intervals]
        ends = [interval[1] for interval in parsed_intervals]
        durations = [end - start for start, end in parsed_intervals]

        # Average duration: Are predictions reasonable lengths?
        metrics["temporal_min_duration"] = float(np.min(durations))
        metrics["temporal_max_duration"] = float(np.max(durations))

        metrics["temporal_avg_duration"] = float(np.mean(durations))
        metrics["temporal_std_duration"] = float(np.std(durations))

        # Start/end point statistics
        metrics["temporal_min_start"] = float(np.min(starts))
        metrics["temporal_max_start"] = float(np.max(starts))
        metrics["temporal_min_end"] = float(np.min(ends))
        metrics["temporal_max_end"] = float(np.max(ends))
        
        metrics["temporal_avg_start"] = float(np.mean(starts))
        metrics["temporal_std_start"] = float(np.std(starts))
        metrics["temporal_avg_end"] = float(np.mean(ends))
        metrics["temporal_std_end"] = float(np.std(ends))

        # Temporal consistency: lower std = more consistent predictions
        metrics["temporal_consistency"] = 1.0 / (
            1.0 + float(np.std(starts)) + float(np.std(ends))
        )

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
        metrics["temporal_min_duration"] = 0.0
        metrics["temporal_max_duration"] = 0.0
        metrics["temporal_min_start"] = 0.0
        metrics["temporal_max_start"] = 0.0
        metrics["temporal_min_end"] = 0.0
        metrics["temporal_max_end"] = 0.0

    return metrics


def test_time_train_metrics(
    solutions: List[str], ground_truth: List[str] = None, task="tag", extra_info=None, distance_threshold=0.10
):
    """
    Compute TTRL metrics using self-consistency (no ground truth required).
    Ground truth parameter is kept for API compatibility but not used.
    """
    assert task == "tag", "Currently only 'tag' task has test-time train metrics implemented"

    model_answers = auto_extract(task, solutions, extra_info=extra_info)
    total = len(model_answers)

    # Use IoU-based clustering for temporal segments
    cluster_labels, parsed_intervals, valid_indices = cluster_temporal_segments(
        model_answers, distance_threshold=distance_threshold
    )

    # Build reward_p: each answer gets reward based on its cluster size
    # For responses which didn't parse, the get 0 reward, which is already set in reward_p initialization
    reward_p = [0.0] * total
    is_valid = [False] * total

    if len(cluster_labels) == 0:
        # No valid intervals, all get zero reward
        entropy = 0.0
        normalized_entropy = 0.0
        estimated_label = ""
        majority_count = 0
        majority_ratio = 0.0
        invalid_counts = Counter(model_answers)
        cluster_counts = Counter(model_answers)
        n_outcomes = len(invalid_counts)
    else:
        # Count cluster memberships
        cluster_counts = Counter(cluster_labels)

        invalid_answers = [ans for i, ans in enumerate(model_answers) if i not in valid_indices]
        invalid_counts = Counter(invalid_answers)

        for i, orig_idx in enumerate(valid_indices):
            cluster_id = cluster_labels[i]
            reward_p[orig_idx] = cluster_counts[cluster_id] / total
            is_valid[orig_idx] = True

        # Compute entropy based on cluster distribution
        entropy = 0.0
        for count in cluster_counts.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log(probability)

        for count in invalid_counts.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log(probability)

        n_outcomes = len(cluster_counts) + len(invalid_counts)

        # Normalize entropy
        if n_outcomes > 1:
            max_entropy = math.log(n_outcomes)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            normalized_entropy = 0.0

        # Find majority cluster and its representative
        majority_cluster_id, majority_count = cluster_counts.most_common(1)[0]

        # Get representative interval from majority cluster (use centroid)
        # majority_intervals = [
        #     parsed_intervals[i]
        #     for i, label in enumerate(cluster_labels)
        #     if label == majority_cluster_id
        # ]
        # avg_start = np.mean([iv[0] for iv in majority_intervals])
        # avg_end = np.mean([iv[1] for iv in majority_intervals])
        # estimated_label = f"({avg_start:.2f}, {avg_end:.2f})"

        majority_ratio = majority_count / total

    rewards_en = [
        (r - (0.75 * normalized_entropy)) if valid else -1.0 
        for r, valid in zip(reward_p, is_valid)
    ]

    ttrl_metrics = {
        "majority_ratio": majority_ratio,
        "normalized_entropy": normalized_entropy,
    }

    temporal_metrics = compute_temporal_metrics(solutions, model_answers)
    ttrl_metrics.update(temporal_metrics)
    ttrl_metrics.update(
        {
            "number_of_clusters": n_outcomes,
            "number_of_valid_clusters": len(cluster_counts),
            "number_of_invalid_clusters": len(invalid_counts),
            "percentage_of_valid_clusters": len(cluster_counts) / n_outcomes if n_outcomes > 0 else 0.0,
            "percentage_of_invalid_clusters": len(invalid_counts) / n_outcomes if n_outcomes > 0 else 0.0,
        }
    )

    return rewards_en, ttrl_metrics


def post_test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    pred_rewards: List,
    task="math",
    extra_info=None,
):
    assert len(solutions) == len(
        ground_truth
    ), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(solutions) == len(
        pred_rewards
    ), f"{len(solutions)} vs {len(pred_rewards)}"
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)

    # counter = Counter(model_answers)

    # true_label_ratio = counter.get(ground_truth, 0) / len(solutions)

    true_rewards, _ = auto_verify(
        task, solutions, [ground_truth] * len(solutions), extra_info=extra_info
    )

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

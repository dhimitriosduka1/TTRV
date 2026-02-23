from collections import defaultdict

from tqdm import tqdm

from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.qwen.qwen_eval import (
    qwen_reward_fn,
    qwen_reward_fn_gpqa,
    simplerl_reward_fn,
    qwen_reward_fn_spatial,
    qwen_reward_fn_temporal,
)


def auto_verify(task, all_outputs, all_labels, extra_info=None):

    task2verify = {
        "math": qwen_reward_fn,
        "simplerl_math": simplerl_reward_fn,
        "gpqa": qwen_reward_fn_gpqa,
        "bbox": qwen_reward_fn_spatial,
        "tag": qwen_reward_fn_temporal,
    }
    assert task in task2verify, f"{task} not in {list(task2verify.keys())}"
    verify_fn = task2verify[task]
    verify_extra_info = defaultdict(list)

    all_outputs = auto_extract(task, all_outputs, extra_info=extra_info)

    rewards = [
        verify_fn(output, label) for output, label in zip(all_outputs, all_labels)
    ]

    verify_extra_info["acc"] = rewards

    verify_extra_info["pred"] = auto_extract(task, all_outputs, extra_info=extra_info)

    return rewards, verify_extra_info

def auto_verify_tag_specific(all_outputs, all_labels, extra_info=None):
    all_outputs = auto_extract("tag", all_outputs, extra_info=extra_info)

    ious = []
    for output, label in zip(all_outputs, all_labels):
        iou = qwen_reward_fn_temporal(output, label)
        ious.append(iou)

    # Compute average IoU
    average_iou = sum(ious) / len(ious) if ious else 0.0

    # Compute IoU at different thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    iou_at_thresholds = {f"iou@{threshold}": (100 * (sum(iou >= threshold for iou in ious) / len(ious))) if ious else 0.0 for threshold in thresholds}

    return {
        "average_iou": average_iou,
        **iou_at_thresholds
    }
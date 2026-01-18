from verl.utils.reward_score.ttrl.qwen.qwen_eval import parse_temporal


def extract_temporal_answer(generated_text):
    """
    Extract temporal interval from generated text.
    Returns string representation of (start, end) for majority voting compatibility.
    """
    interval = parse_temporal(generated_text)
    if interval is None:
        return ""

    return f"({interval[0]:.2f}, {interval[1]:.2f})"

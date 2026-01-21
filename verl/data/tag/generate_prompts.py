import json
import pickle

VIDEO_BASE = "/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec"
PICKLE_PATH = "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.pkl"
CHUNK_DURATION = 15
PROMPT_TEMPLATE = """TASK: Temporal localization in egocentric video.

    ACTION TO LOCATE: "{caption}"

    SEED WINDOW (approximate): {seed_start:.2f}s to {seed_end:.2f}s (use as starting point only, may be inaccurate).

    ANALYSIS STEPS:
    1. Watch the video and identify the camera wearer's hands throughout.
    2. Find when the described action STARTS: the exact moment of first intentional movement toward the action (hand reaches, begins grasp, or object starts moving due to wearer).
    3. Find when the action ENDS: the exact moment the action goal is achieved (object released/placed, hands withdraw, result is stable).

    VISUAL CUES TO TRACK:
    - Hand position and motion relative to target objects
    - Object state changes (picked up, moved, opened, closed, placed)
    - Contact events (hand touches object, object touches surface)
    - Motion ends (object comes to rest, hand stops moving)

    CRITICAL RULES:
    - Boundaries must be TIGHT: start at first evidence of action, end when action completes
    - Do NOT include preparation or aftermath unless part of the described action
    - If action spans multiple sub-actions, include the full sequence
    - Times are relative to video start (0.0s = first frame)

    OUTPUT FORMAT: [start_seconds, end_seconds]
    Times should be relative to the start of the first video segment.
    Example: [2.5, 18.8]

    Your answer:
"""


def get_chunk_paths(video_id: str, start_time: float, end_time: float):
    """Get all 15-second chunk video paths covering start-padding to end+padding.

    Args:
        video_id: EGO4D video identifier
        start_time: Action start time in seconds
        end_time: Action end time in seconds

    Returns:
        Tuple of (list of video dicts, context_start_time)
    """
    context_start = start_time
    context_end = end_time

    start_chunk_idx = int(context_start // CHUNK_DURATION)
    end_chunk_idx = int(context_end // CHUNK_DURATION)

    chunks = []
    for chunk_idx in range(start_chunk_idx, end_chunk_idx + 1):
        chunk_start_time = chunk_idx * CHUNK_DURATION
        chunk_path = f"file://{VIDEO_BASE}/{video_id}.mp4/{chunk_start_time}.mp4"
        chunks.append(chunk_path)

    # context_start is the time offset from the first chunk's start
    first_chunk_start = start_chunk_idx * CHUNK_DURATION
    return chunks, first_chunk_start


def process_items(items, start_idx=0):
    """Convert list of items to format expected by TTRL training.

    Args:
        items: List of (video_id, start, end, caption) tuples
        start_idx: Starting index for extra_info

    Returns:
        List of training data dicts

    """
    training_data = []
    for idx, item in enumerate(items):
        video_id, start, end, caption = item
        start, end = float(start), float(end)

        chunks, first_chunk_start = get_chunk_paths(video_id, start, end)

        if len(chunks) == 0:
            print(f"Warning: No chunks found for video_id={video_id}, skipping")
            continue

        # Timestamps relative to first chunk start
        rel_start = start - first_chunk_start
        rel_end = end - first_chunk_start

        assert rel_start >= 0 and rel_end >= 0 and rel_start < rel_end

        # This has to be a simple dict as follows
        training_data.append(
            {
                "prompt": "".join(["<video>"] * len(chunks))
                + PROMPT_TEMPLATE.format(
                    caption=caption, seed_start=rel_start, seed_end=rel_end
                ),
                "video_paths": chunks,
                "answer": [0.0, 0.0],
                "source": "dduka",
                "id": idx,
            }
        )

    return training_data


def main():
    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)

    train_items = data[:10]
    test_items = data[100:110]

    train_data = process_items(train_items, start_idx=0)
    test_data = process_items(test_items, start_idx=0)

    with open("train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open("test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Created train.json with {len(train_data)} items")
    print(f"Created test.json with {len(test_data)} items")
    print(f"\nSample train item:\n{json.dumps(train_data[0], indent=2)}")


if __name__ == "__main__":
    main()

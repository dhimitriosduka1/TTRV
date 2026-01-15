#!/usr/bin/env python3
"""Generate training/test examples from EGO4D pickle for TTRL."""
import pickle
import json

VIDEO_BASE = "/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec"
PICKLE_PATH = "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.pkl"
CHUNK_DURATION = 15  # seconds per chunk
CONTEXT_PADDING = 10  # seconds of context before/after action

PROMPT_TEMPLATE = """<video>
TASK: Locate when "{caption}" occurs in this video.

HINT: The action might be near {seed_start:.1f}s to {seed_end:.1f}s.

OUTPUT FORMAT: [start_seconds, end_seconds]
Example: [2.5, 4.8]

Your answer:"""


def get_chunk_paths(video_id, start_time, end_time):
    """Get all 15-second chunk video paths covering start-10s to end+10s."""
    context_start = max(0, start_time - CONTEXT_PADDING)
    context_end = end_time + CONTEXT_PADDING

    start_chunk = int(context_start // CHUNK_DURATION)
    end_chunk = int(context_end // CHUNK_DURATION)

    # Return list of video dicts in Qwen VL format
    chunks = []
    for chunk_idx in range(start_chunk, end_chunk + 1):
        chunk_start_time = chunk_idx * CHUNK_DURATION
        chunk_path = f"file://{VIDEO_BASE}/grp-{video_id}.mp4/{chunk_start_time}.mp4"
        chunks.append({"video": chunk_path, "nframes": 32})
        break

    return chunks, context_start


def process_items(items, start_idx=0):
    """Convert list of items to format expected by TTRL training."""
    training_data = []
    for idx, item in enumerate(items):
        video_id, start, end, caption = item
        start, end = float(start), float(end)

        chunks, context_start = get_chunk_paths(video_id, start, end)

        # Timestamps relative to context window
        rel_start = start - context_start
        rel_end = end - context_start

        prompt_text = PROMPT_TEMPLATE.format(
            caption=caption, seed_start=rel_start, seed_end=rel_end
        )

        # Format matching tag/train.json structure:
        # - "prompt": list of chat messages (Qwen format)
        # - "videos": list of video dicts with "video" and "nframes" keys
        # - "answer": ground truth in [start, end] format
        # - "data_source": "EGO4D" (maps to "tag" task in ttrl.py)
        # - "reward_model": dict with style and ground_truth
        # - "extra_info": metadata including index and video_id
        training_data.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "videos": chunks,
                "answer": "[0, 0]",  # Placeholder - actual ground truth would go here
                "data_source": "EGO4D",
                "reward_model": {"style": "rule", "ground_truth": "[0, 0]"},
                "extra_info": {"index": start_idx + idx, "video_id": video_id},
            }
        )

    return training_data


def main():
    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)

    # Train: first 10 items, Test: items 100-110 (different samples)
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

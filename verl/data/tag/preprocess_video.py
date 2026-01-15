#!/usr/bin/env python3
"""Preprocess video JSON files to Parquet format for TTRL training.

This script is specifically designed for video datasets (like EGO4D) 
where the JSON already contains the correct format with:
- "prompt": list of chat messages
- "videos": list of video dicts with "video" and "nframes" keys
- "answer": ground truth
- "data_source": dataset identifier
- "reward_model": dict with style and ground_truth
- "extra_info": metadata

Usage:
    python preprocess_video.py
"""
import os
import datasets


def make_map_fn(split):
    """Create a mapping function that preserves video fields."""
    def process_fn(example, idx):
        # The JSON already has the correct format, just ensure all fields are present
        data = {
            "data_source": example.get("data_source", "EGO4D"),
            "prompt": example["prompt"],  # Already in chat format
            "videos": example["videos"],  # List of video dicts
            "ability": "temporal_grounding",
            "reward_model": example.get("reward_model", {
                "style": "rule",
                "ground_truth": example.get("answer", "[0, 0]")
            }),
            "extra_info": {
                **example.get("extra_info", {}),
                "split": split,
            },
        }
        return data

    return process_fn


if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load JSON files
    train_path = os.path.join(data_dir, 'train.json')
    test_path = os.path.join(data_dir, 'test.json')
    
    train_dataset = datasets.load_dataset("json", data_files=train_path, split='train')
    test_dataset = datasets.load_dataset("json", data_files=test_path, split='train')

    # Apply preprocessing
    train_dataset = train_dataset.map(
        function=make_map_fn("train"), 
        with_indices=True,
        remove_columns=["answer"]  # answer is moved to reward_model.ground_truth
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test"), 
        with_indices=True,
        remove_columns=["answer"]
    )

    # Save to Parquet
    train_output = os.path.join(data_dir, 'train.parquet')
    test_output = os.path.join(data_dir, 'test.parquet')
    
    train_dataset.to_parquet(train_output)
    test_dataset.to_parquet(test_output)
    
    print(f"Created {train_output} with {len(train_dataset)} items")
    print(f"Created {test_output} with {len(test_dataset)} items")
    
    # Show sample
    print("\nSample train item:")
    sample = train_dataset[0]
    for key, value in sample.items():
        print(f"  {key}: {value}")

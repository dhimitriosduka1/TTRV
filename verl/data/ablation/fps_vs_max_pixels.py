import os
import datasets

# Define the three ablation runs to maintain a constant token budget (~3,200 tokens/sec)
ABLATION_RUNS = {
    "run1_temporal": {"max_pixels": 120000, "fps": 12},
    "run2_spatial": {"max_pixels": 360000, "fps": 4},
    "run3_balanced": {"max_pixels": 180000, "fps": 8},
}

def make_map_fn(split, fps, max_pixels):
    """Create a mapping function that injects specific ablation settings."""

    def process_fn(example, idx):
        data = {
            "data_source": example.get("data_source", "EGO4D"),
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "ability": "temporal_grounding",
            "reward_model": {"style": "rule", "ground_truth": example["answer"]},
            "extra_info": {
                "split": split,
                "index": f"{example.get('data_source', 'EGO4D')}-{idx}",
                "uuid": example.get("uuid"),
            },
            "videos": [
                {"video": path, "fps": fps, "max_pixels": max_pixels}
                for path in example["video_paths"]
            ],
        }
        return data

    return process_fn


if __name__ == "__main__":
    data_dir = "/u/dduka/project/RL/TTRV/verl/data/tag/"

    train_path = os.path.join(data_dir, "train.json")
    test_path = os.path.join(data_dir, "test.json")

    # Load the base datasets once
    raw_train_dataset = datasets.load_dataset("json", data_files=train_path, split="train")
    raw_test_dataset = datasets.load_dataset("json", data_files=test_path, split="train")

    # Iterate through the ablation configurations
    for run_name, config in ABLATION_RUNS.items():
        print(f"\n --- Generating {run_name} (FPS: {config['fps']}, Max Pixels: {config['max_pixels']}) ---")
        
        # Map with the specific config
        mapped_train = raw_train_dataset.map(
            function=make_map_fn("train", config["fps"], config["max_pixels"]),
            with_indices=True,
            desc=f"Mapping train for {run_name}"
        )
        mapped_test = raw_test_dataset.map(
            function=make_map_fn("test", config["fps"], config["max_pixels"]),
            with_indices=True,
            desc=f"Mapping test for {run_name}"
        )

        # Output to distinct parquet files
        train_output = os.path.join(data_dir, f"train_{run_name}_scaled.parquet")
        test_output = os.path.join(data_dir, f"test_{run_name}_scaled.parquet")

        mapped_train.to_parquet(train_output)
        mapped_test.to_parquet(test_output)

        print(f"Created {train_output} with {len(mapped_train)} items")
        print(f"Created {test_output} with {len(mapped_test)} items")
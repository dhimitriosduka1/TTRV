import os
import datasets

DEFAULT_MAX_PIXELS = 360 * 420
DEFAULT_FPS = 8


def make_map_fn(split):
    """Create a mapping function that preserves video fields."""

    def process_fn(example, idx):
        data = {
            "data_source": example.get("data_source", "EGO4D"),
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "ability": "temporal_grounding",
            "reward_model": {"style": "rule", "ground_truth": example["answer"]},
            "extra_info": {
                "split": split,
                "index": f"{example.get("data_source", "EGO4D")}-{idx}",
            },
            "videos": [
                {"video": path, "fps": DEFAULT_FPS, "max_pixels": DEFAULT_MAX_PIXELS}
                for path in example["video_paths"]
            ],
        }
        print(data)
        return data

    return process_fn


if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(data_dir, "train.json")
    test_path = os.path.join(data_dir, "test.json")

    train_dataset = datasets.load_dataset("json", data_files=train_path, split="train")
    test_dataset = datasets.load_dataset("json", data_files=test_path, split="train")

    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
    )
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_output = os.path.join(data_dir, "train.parquet")
    test_output = os.path.join(data_dir, "test.parquet")

    train_dataset.to_parquet(train_output)
    test_dataset.to_parquet(test_output)

    print(f"Created {train_output} with {len(train_dataset)} items")
    print(f"Created {test_output} with {len(test_dataset)} items")

    print("\nSample train item:")
    sample = train_dataset[0]
    for key, value in sample.items():
        print(f"  {key}: {value}")

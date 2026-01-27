import json
from torch.utils.data import DataLoader, Subset, random_split
from utils.utils import set_seed


def init_dataloaders(datasets, val_ratio=0.2, batch_size=32, info_file_path=None):

    print("Prepare DataLoaders...")

    dataset, test_dataset = datasets

    classes = test_dataset.classes

    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size

    set_seed()
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if info_file_path is not None:
        splits = {
            "train_indices": [
                {"index": idx, "class": dataset[idx][1]}
                for idx in train_dataset.indices
            ],
            "val_indices": [
                {"index": idx, "class": dataset[idx][1]} for idx in val_dataset.indices
            ],
            "test_indices": [
                {"index": idx, "class": test_dataset[idx][1]}
                for idx in range(len(test_dataset))
            ],
        }

        with open(info_file_path, "w") as f:
            json.dump(splits, f)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Done preparing DataLoaders.")

    return train_loader, val_loader, test_loader, classes


def update_splits_after_unlearning(data_splits_file, unlearn_samples_file, output_file):

    # Load data splits and unlearn indices
    with open(data_splits_file, "r") as f:
        splits = json.load(f)
    with open(unlearn_samples_file, "r") as f:
        unlearn_indices = json.load(f)

    # Extract unlearn indices
    unlearn_indices_set = {entry["index"] for entry in unlearn_indices}

    # Update splits
    updated_splits = {
        "train_indices": [
            entry
            for entry in splits["train_indices"]
            if entry["index"] not in unlearn_indices_set
        ],
        "val_indices": [
            entry
            for entry in splits["val_indices"]
            if entry["index"] not in unlearn_indices_set
        ],
        "test_indices": splits["test_indices"],  # Test set remains unchanged
    }

    # Save updated splits
    with open(output_file, "w") as f:
        json.dump(updated_splits, f)

    print(f"Updated splits saved to {output_file}")


def recreate_dataloaders(data_splits_file, datasets, batch_size=32):

    print("Recreating DataLoaders...")

    # Load updated splits
    with open(data_splits_file, "r") as f:
        splits = json.load(f)

    dataset, test_dataset = datasets
    classes = test_dataset.classes

    # Extract indices
    train_indices = [entry["index"] for entry in splits["train_indices"]]
    val_indices = [entry["index"] for entry in splits["val_indices"]]
    test_indices = [entry["index"] for entry in splits["test_indices"]]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    if not (
        len(test_indices) == len(test_dataset)
        and all(idx == i for i, idx in enumerate(test_indices))
    ):
        test_dataset = Subset(test_dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Done recreating DataLoaders.")

    return train_loader, val_loader, test_loader, classes

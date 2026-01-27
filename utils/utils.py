import json
import torch
import random
import numpy as np

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    np.random.seed(seed)

    random.seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def select_samples_to_unlearn(
    data_splits_file, unlearn_samples_output_file, unlearn_ratio=0.1, target_class=None,
):

    # Load data splits
    with open(data_splits_file, "r") as f:
        splits = json.load(f)

    # Combine train and validation indices
    combined_indices = splits["train_indices"] + splits["val_indices"]

    if target_class is not None:
        if not isinstance(target_class, (list, set, tuple)):
            target_class = [target_class]
        combined_indices = [
            entry for entry in combined_indices if entry["class"] in target_class
        ]

    set_seed()
    unlearn_count = int(unlearn_ratio * len(combined_indices))
    unlearn_indices = random.sample(combined_indices, unlearn_count)

    # Save unlearn indices
    with open(unlearn_samples_output_file, "w") as f:
        json.dump(unlearn_indices, f)

    print(f"Unlearn indices saved to {unlearn_samples_output_file}")


def select_targeted_samples_to_unlearn(
    data_splits_file, unlearn_samples_output_file, targeted_samples
):

    if isinstance(targeted_samples, int):
        targeted_samples = [targeted_samples]

    # Load data splits
    with open(data_splits_file, "r") as f:
        splits = json.load(f)

    # Combine train and validation indices
    combined_indices = splits["train_indices"] + splits["val_indices"]

    # Filter to retain only targeted samples, preserving their metadata
    valid_unlearn_samples = [
        entry for entry in combined_indices if entry["index"] in targeted_samples
    ]

    # Save the valid unlearn indices
    with open(unlearn_samples_output_file, "w") as f:
        json.dump(valid_unlearn_samples, f)

    print(f"Unlearn indices (targeted) saved to {unlearn_samples_output_file}")


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

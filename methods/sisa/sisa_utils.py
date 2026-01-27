import json
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from utils.utils import set_seed


def create_sisa_structure(dataset, shards=3, slices_per_shard=5, results_path="."):
    """
    Create the SISA structure for the MNIST dataset using random_split.
    - 3 shards, each with 5 slices.
    - Save indices and classes for each sample.

    Args:
        dataset: PyTorch dataset object (MNIST training set).
        shards: Number of shards (default: 3).
        slices_per_shard: Number of slices per shard (default: 5).

    Returns:
        sisa_structure: Dictionary representing the SISA structure.
    """
    # Total size of the dataset

    shard_size = len(dataset) // shards

    shard_sizes = [shard_size] * shards

    shard_sizes[-1] += len(dataset) % shards

    # Split dataset into shards
    set_seed()
    shards = random_split(dataset, shard_sizes)
    sisa_structure = {}

    for shard_id, shard in enumerate(shards):

        shard_indices = shard.indices

        # Get the size of each slice within the shard
        slice_size = len(shard) // slices_per_shard

        slice_sizes = [slice_size] * slices_per_shard

        slice_sizes[-1] += len(shard) % slices_per_shard

        # Split the shard into slices
        set_seed()
        slices = random_split(shard, slice_sizes)
        sisa_structure[f"shard_{shard_id}"] = {}

        # Save indices and classes for each slice
        for slice_id, slice_data in enumerate(slices):

            slice_indices = [shard_indices[idx] for idx in slice_data.indices]

            slice_classes = [dataset[idx][1] for idx in slice_indices]

            sisa_structure[f"shard_{shard_id}"][f"slice_{slice_id}"] = {
                "indices": slice_indices,
                "classes": slice_classes,
            }

    with open(results_path, "w") as f:
        json.dump(sisa_structure, f, indent=4)
    print(f"SISA structure saved to {results_path}")


def recreate_sisa_dataloaders(datasets, info_file_path, batch_size=32, val_ratio=0.1):
    """
    Recreates the SISA structure from a JSON file and prepares DataLoaders for each slice.
    Splits each slice into 90% training and 10% validation data.

    Args:
        json_file (str): Path to the JSON file containing the SISA structure.
        dataset (Dataset): The original dataset to recreate slices from.
        batch_size (int): Batch size for the DataLoaders.
        val_ratio (float): Proportion of each slice to use as the validation set.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary with the structure {shard -> slice -> {"train": DataLoader, "val": DataLoader}}.
    """
    # Load the SISA structure

    dataset, test_dataset = datasets

    classes = test_dataset.classes

    with open(info_file_path, "r") as f:
        sisa_structure = json.load(f)

    dataloaders = {}

    # Iterate over shards and slices
    for shard_id, shard_data in sisa_structure.items():
        dataloaders[shard_id] = {}

        for slice_id, slice_data in shard_data.items():
            # Extract indices for this slice
            slice_indices = slice_data["indices"]

            # Create a subset from the dataset
            slice_subset = Subset(dataset, slice_indices)

            # Split into training and validation sets
            val_size = int(len(slice_subset) * val_ratio)
            train_size = len(slice_subset) - val_size
            train_subset, val_subset = random_split(
                slice_subset, [train_size, val_size]
            )

            # Create DataLoaders
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # Store DataLoaders in the structure
            dataloaders[shard_id][slice_id] = {"train": train_loader, "val": val_loader}

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataloaders["test"] = test_loader

    return dataloaders, classes


def update_sisa_structure(
    unlearn_samples_path,
    sisa_structure_path,
    updated_structure_path,
    deleted_samples_path,
):
    """
    Handles unlearning by identifying affected shards and slices, updating SISA structure,
    and triggering retraining and testing.

    Args:
        unlearn_samples_path (str): Path to JSON file with samples to unlearn.
        sisa_structure_path (str): Path to SISA structure JSON file.
        updated_structure_path (str): Path to save updated SISA structure JSON file.
        deleted_samples_path (str): Path to save deleted samples JSON file.
    """
    # Load samples to unlearn
    with open(unlearn_samples_path, "r") as f:
        unlearn_samples = json.load(f)

    # Load the SISA structure
    with open(sisa_structure_path, "r") as f:
        sisa_structure = json.load(f)

    # Track affected shards, slices, and samples to delete
    affected_shards = {}
    deleted_samples = []

    # Identify affected shards and slices
    for sample in unlearn_samples:
        index, label = sample["index"], sample["class"]
        for shard, slices in sisa_structure.items():
            for slice_name, slice_data in slices.items():
                if index in slice_data["indices"] and label in slice_data["classes"]:
                    # Track the lowest affected slice
                    if shard not in affected_shards:
                        affected_shards[shard] = []
                    affected_shards[shard].append(slice_name)

                    # Remove the sample from the slice
                    idx_position = slice_data["indices"].index(index)
                    slice_data["indices"].pop(idx_position)
                    slice_data["classes"].pop(idx_position)

                    # Track deleted samples
                    deleted_samples.append(sample)
                    break  # Move to the next sample after finding a match

    # Deduplicate and sort slice flags
    affected_shards = {
        shard: sorted(set(affected_shards[shard])) for shard in sorted(affected_shards)
    }

    # Save the updated SISA structure
    with open(updated_structure_path, "w") as f:
        json.dump(sisa_structure, f)

    # Save the deleted samples
    with open(deleted_samples_path, "w") as f:
        json.dump(deleted_samples, f)

    # Print retraining plan
    print("Retraining Plan:")
    for shard, slices in affected_shards.items():
        print(f"  Shard: {shard}, Start from Slice: {slices[0]} onward")

    return affected_shards


def calculate_shard_metrics(true_labels, shard_predictions):
    """
    Calculate metrics for a single shard's predictions.

    Args:
        true_labels (list): True labels for the dataset.
        shard_predictions (list): Predictions from a shard model.

    Returns:
        dict: A dictionary with accuracy, precision, recall, and F1 score.
    """
    accuracy = accuracy_score(true_labels, shard_predictions)
    precision = precision_score(
        true_labels, shard_predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        true_labels, shard_predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(true_labels, shard_predictions, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def weighted_voting(true_labels, predictions, weights):
    """
    Perform weighted voting aggregation of shard predictions.

    Args:
        true_labels (list): True labels for the dataset.
        predictions (list): List of shard predictions (each a list of predictions for samples).
        weights (list): List of weights (accuracies) for each shard.

    Returns:
        list: Aggregated predictions.
    """
    num_samples = len(true_labels)
    num_shards = len(predictions)

    # Create an array to store shard predictions for each sample
    shard_preds = np.array(predictions)  # Shape: (num_shards, num_samples)
    weights = np.array(weights).reshape(-1, 1)  # Shape: (num_shards, 1)

    # Weighted voting: Take the weighted mode of predictions
    weighted_votes = np.zeros((10, num_samples))  # Assuming 10 classes (MNIST)
    for shard_idx in range(num_shards):
        for i, pred in enumerate(shard_preds[shard_idx]):
            weighted_votes[pred, i] += weights[shard_idx]

    # Final prediction: Class with the highest weighted vote
    aggregated_predictions = np.argmax(weighted_votes, axis=0)
    return aggregated_predictions.tolist()


def evaluate_aggregated_model(results, classes):
    """
    Evaluate aggregated predictions and print metrics, confusion matrix.

    Args:
        results (dict): Results dictionary containing shard-specific predictions and true_labels.
    """
    # Initialize variables for aggregated predictions and shard metrics
    shard_metrics = {}
    shards_accuracies = []
    shards_predictions = []
    true_labels = None

    # Evaluate each shard
    for shard_id, shard_data in results.items():
        shard_preds = shard_data["predictions"]
        shard_true_labels = shard_data["true_labels"]

        # Ensure true_labels are consistent across shards
        if true_labels is None:
            true_labels = shard_true_labels
        elif true_labels != shard_true_labels:
            raise ValueError(
                f"True labels in shard {shard_id} do not match other shards!"
            )

        # Calculate metrics for the shard
        metrics = calculate_shard_metrics(true_labels, shard_preds)
        shard_metrics[shard_id] = metrics
        shards_accuracies.append(metrics["accuracy"])
        shards_predictions.append(shard_preds)
        print(f"Shard {shard_id} Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")

    aggregated_predictions = weighted_voting(
        true_labels=true_labels,
        predictions=shards_predictions,
        weights=shards_accuracies,
    )

    # Calculate metrics for the aggregated predictions
    aggregated_metrics = calculate_shard_metrics(true_labels, aggregated_predictions)
    print("\nAggregated Model Metrics:")
    print(f"  Accuracy: {aggregated_metrics['accuracy']:.4f}")
    print(f"  Precision: {aggregated_metrics['precision']:.4f}")
    print(f"  Recall: {aggregated_metrics['recall']:.4f}")
    print(f"  F1 Score: {aggregated_metrics['f1_score']:.4f}")

    # Generate and display confusion matrix
    cm = confusion_matrix(true_labels, aggregated_predictions)
    plt.figure(figsize=(10, 8))
    sb.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix of Aggregated Model")
    plt.show()

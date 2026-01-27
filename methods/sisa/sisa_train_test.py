import os
import json
import numpy as np
from pathlib import Path

import torch.nn as nn

from utils.train_test_metrics import train_model, test_model
from utils.utils import save_model

MULTI_GPU = False


def sisa_train(
    dataloaders,
    num_epochs,
    save_models_metrics_dir,
    init_model_func,
    learning_rate=0.001,
):

    save_path = Path(save_models_metrics_dir)
    save_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Iterate over each shard
    for shard_id, slices in dataloaders.items():
        if shard_id == "test":  # Skip the test loader
            continue

        print(f"Training shard: {shard_id}")

        # Initialize a new model for the shard
        model, model_name, criterion, optimizer, _ = init_model_func(
            learning_rate=learning_rate
        )
        if MULTI_GPU:
            model = nn.DataParallel(model, device_ids=[0, 1])

        # Iterate over slices in the shard
        for slice_id, loaders in slices.items():

            print(f"  Training slice: {slice_id}")

            # Get train and validation loaders for this slice
            train_loader = loaders["train"]
            val_loader = loaders["val"]

            slice_model_name = f"{shard_id}_{slice_id}_" + model_name

            # Call the slice-level training function
            train_model(
                model=model,
                model_name=slice_model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                results_path=save_path
            )

        shard_model_path = save_path / f"{shard_id}_final_model.pth"
        save_model(model, shard_model_path)
        print(f"Saved final shard model to {shard_model_path}")


def sisa_test(
    dataloaders, saved_models_metrics_dir, init_model_func, clear_solo_models_preds=True, results_path=Path(".")
):
    test_loader = dataloaders["test"]

    # Initialize the evaluation results dictionary
    evaluation_results = {}

    for shard_id in [key for key in dataloaders.keys() if key != "test"]:

        # Path to the final model for this shard
        shard_model_path = f"{saved_models_metrics_dir}/{shard_id}_final_model.pth"

        # Load model
        model, model_name, *_ = init_model_func()
        if MULTI_GPU:
            model = nn.DataParallel(model, device_ids=[0, 1])

        shard_model_name = f"{saved_models_metrics_dir}/{shard_id}_" + model_name

        # Call the evaluation function
        test_model(model, shard_model_name, shard_model_path, test_loader, results_path)

        intermediate_json_path = f"{shard_model_name}_predictions.json"

        # Load intermediate predictions JSON
        with open(intermediate_json_path, "r") as f:
            shard_data = json.load(f)

        # Add shard-specific predictions and true labels to evaluation_results
        evaluation_results[shard_id] = {
            "predictions": np.array(shard_data["predictions"])
            .flatten()
            .astype(int)
            .tolist(),
            "true_labels": np.array(shard_data["true_labels"])
            .flatten()
            .astype(int)
            .tolist(),
        }

        # Delete intermediate JSON if clear_solo_models_preds is True
        if clear_solo_models_preds:
            os.remove(intermediate_json_path)

    # Save predictions and true labels to a JSON file
    output_path = results_path / "sisa_final_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(evaluation_results, f)

    print(f"Evaluation results saved to {output_path}")


def retrain_sisa_framework(
    dataloaders,
    affected_shards,
    num_epochs,
    save_models_metrics_dir,
    init_model_func,
    learning_rate,
):
    """
    Retrain the SISA framework starting from the flagged slices for affected shards.

    Args:
        dataloaders (dict): Dataloaders for the updated SISA structure.
        affected_shards (dict): Dictionary of affected shards and their flagged slices.
        num_epochs (int): Number of epochs for training each slice.
        save_models_metrics_dir (str): Directory to save the updated models and metrics.
    """
    save_path = Path(save_models_metrics_dir)
    save_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Iterate over affected shards
    for shard_id, flagged_slices in affected_shards.items():
        print(f"Retraining shard: {shard_id}")

        # Initialize a new model for this shard
        model, model_name, criterion, optimizer, _ = init_model_func(
            learning_rate=learning_rate
        )
        if MULTI_GPU:
            model = nn.DataParallel(model, device_ids=[0, 1])

        # Iterate over slices in the shard
        for slice_id, loaders in dataloaders[shard_id].items():
            current_slice_idx = int(slice_id.split("_")[1])

            # Strip the prefix "slice_" from flagged_slices[0] before converting to int
            flagged_slice_idx = int(flagged_slices[0].split("_")[1])

            # Only retrain starting from flagged slices
            if current_slice_idx >= flagged_slice_idx:
                print(f"  Retraining slice: {slice_id}")

                # Get train and validation loaders for this slice
                train_loader = loaders["train"]
                val_loader = loaders["val"]

                slice_model_name = f"{shard_id}_{slice_id}_" + model_name

                # Train the model on this slice
                train_model(
                    model=model,
                    model_name=slice_model_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=num_epochs,
                    results_path=save_path,
                )

        # Save the final model for the shard
        shard_model_path = save_path / f"{shard_id}_final_model.pth"
        save_model(model, shard_model_path)
        print(f"Saved updated model for {shard_id} to {shard_model_path}")

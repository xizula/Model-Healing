import math
import torch
import json
from utils.utils import DEVICE
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset


def compute_gradient_on_subset(model, criterion, dataset_subset, batch_size):
    """
    Compute the average gradient Δ_rem = ∇L(θ, D') over the given dataset_subset.
    """
    dataloader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False)

    grad_dict = {}
    total_samples = 0

    model.train()
    for inputs, targets in tqdm(dataloader, desc="Computing gradients"):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        current_batch = inputs.size(0)
        total_samples += current_batch
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in grad_dict:
                    grad_dict[name] = param.grad.detach().clone() * current_batch
                else:
                    grad_dict[name] += param.grad.detach() * current_batch

    # Average gradients over the entire subset
    for name in grad_dict:
        grad_dict[name] /= total_samples

    return grad_dict


def compute_fisher_on_subset(model, criterion, dataset_subset, batch_size):
    """
    Compute a diagonal approximation of the Fisher Information Matrix F over the given dataset_subset.
    It averages the squared gradients.
    """
    dataloader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False)
    fisher_diag = {}
    total_samples = 0

    model.eval()
    for inputs, targets in tqdm(dataloader, desc="Computing Fisher"):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        current_batch = inputs.size(0)
        total_samples += current_batch
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in fisher_diag:
                    fisher_diag[name] = (param.grad.detach() ** 2) * current_batch
                else:
                    fisher_diag[name] += (param.grad.detach() ** 2) * current_batch

    for name in fisher_diag:
        fisher_diag[name] /= total_samples

    return fisher_diag


def remove_from_fisher_incrementally(
    fisher_diag, model, criterion, dataset_removed, batch_size
):
    dataloader = DataLoader(dataset_removed, batch_size=batch_size, shuffle=False)
    total_removed_samples = 0

    model.eval()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        batch_samples = inputs.size(0)
        total_removed_samples += batch_samples
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_diag[name] -= (param.grad.detach() ** 2) * batch_samples

    total_samples_remaining = fisher_diag["_total_samples"] - total_removed_samples
    for name in fisher_diag:
        if name != "_total_samples":
            fisher_diag[name] = torch.clamp(fisher_diag[name], min=1e-8)
            fisher_diag[name] /= total_samples_remaining
    fisher_diag["_total_samples"] = total_samples_remaining

    return fisher_diag


def iterative_fisher_unlearn(
    model,
    criterion,
    full_dataset,
    removal_indices,
    sigma,
    deletion_batch_size,
    compute_batch_size,
    eps,
    max_norm,
):
    """
    Implements the iterative Fisher unlearning procedure following theory:

    Inputs:
      - model: a pretrained PyTorch model (trained on full dataset D).
      - criterion: loss function (e.g., CrossEntropyLoss).
      - full_dataset: the full training dataset D (e.g., MNIST training set).
      - removal_indices: list of indices (from the JSON file) to be deleted (Dₘ). E.g., 6000 samples.
      - sigma: noise parameter σ.
      - deletion_batch_size: desired mini-batch size for deletion (m′). E.g., 1000.
      - compute_batch_size: batch size used when computing gradients/Fisher (BATCH_SIZE).
      - eps: for numerical stability

    Procedure:
      1. Let current_indices = set(range(len(full_dataset))).
      2. Partition removal_indices into mini-batches of size deletion_batch_size.
      3. For each mini-batch, update current_indices by removing those indices.
      4. Create a Subset from full_dataset using current_indices (this is D').
      5. Compute Δ_rem and diagonal Fisher F on D' and update model:
             θ ← θ − F⁻¹ Δ_rem + σ · F^(–1/4) · ε.
    """
    full_size = len(full_dataset)
    current_indices = set(range(full_size))

    # Partition removal_indices into mini-batches, where s = m /m'
    removal_list = list(removal_indices)
    num_batches = math.ceil(len(removal_list) / deletion_batch_size)
    partitioned_removals = [
        removal_list[i * deletion_batch_size : (i + 1) * deletion_batch_size]
        for i in range(num_batches)
    ]
    print(
        f"Total deletion samples: {len(removal_list)}; partitioned into {num_batches} mini-batches (each up to {deletion_batch_size} samples)."
    )

    # Iterate over each deletion mini-batch
    for i, batch in enumerate(
        tqdm(partitioned_removals, desc="Fisher step over mini-batches")
    ):
        # Remove the current batch of indices from current_indices
        current_indices -= set(batch)
        updated_indices = sorted(list(current_indices))
        # Create a Subset corresponding to the updated dataset D' = D \ (deleted so far)
        dataset_remaining = Subset(full_dataset, updated_indices)
        print(
            f"Iteration {i+1}/{num_batches}: Remaining dataset size = {len(dataset_remaining)}"
        )

        # Compute the average gradient and diagonal Fisher on D'
        grad_dict = compute_gradient_on_subset(
            model, criterion, dataset_remaining, compute_batch_size
        )
        fisher_diag = compute_fisher_on_subset(
            model, criterion, dataset_remaining, compute_batch_size
        )

        # Update model parameters using the Newton correction and noise injection
        with torch.no_grad():
            for name in grad_dict:
                grad = grad_dict[name]
                norm = grad.norm(2).item()
                grad_min = grad.min().item()
                grad_max = grad.max().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                print(
                    f"[Raw] Param {name}: norm = {norm:.4e}, min = {grad_min:.4e}, max = {grad_max:.4e}, mean = {grad_mean:.4e}, std = {grad_std:.4e}"
                )

            # First, compute and clip gradients, and monitor norms
            total_grad_norm_before = 0.0
            total_grad_norm_after = 0.0
            for name in grad_dict:
                norm_before = grad_dict[name].norm(2)
                total_grad_norm_before += norm_before.item()
                if norm_before > max_norm:
                    grad_dict[name] = grad_dict[name] * (max_norm / norm_before)
                norm_after = grad_dict[name].norm(2)
                total_grad_norm_after += norm_after.item()

            print(
                f"Iteration {i+1}: Total gradient norm before clipping = {total_grad_norm_before:.4f}"
            )
            print(
                f"Iteration {i+1}: Total gradient norm after clipping  = {total_grad_norm_after:.4f}"
            )

            # Now, update model parameters using the clipped gradients and monitor the Newton update norm
            total_update_norm = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad:

                    inv_fisher = (fisher_diag[name] + eps).pow(-1)
                    newton_update = inv_fisher * grad_dict[name]
                    total_update_norm += newton_update.norm(2).item()
                    param.data = param.data - newton_update

                    inv_fisher_quarter = (fisher_diag[name] + eps).pow(-0.25)
                    noise = torch.randn_like(param.data)
                    param.data = param.data + sigma * inv_fisher_quarter * noise

            print(
                f"Iteration {i+1}: Total Newton update norm = {total_update_norm:.4f}"
            )
        print(f"Iteration {i+1}/{num_batches} update completed.")

    return model


def create_unlearning_dataloader(unlearn_file, dataset, batch_size=32):
    
    with open(unlearn_file, "r") as f:
        unlearn_samples = json.load(f)

    unlearn_indices = [entry["index"] for entry in unlearn_samples]

    unlearn_dataset = Subset(dataset, unlearn_indices)

    unlearn_loader = DataLoader(unlearn_dataset, batch_size=batch_size, shuffle=False)
    
    return unlearn_indices, unlearn_loader

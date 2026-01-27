import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm, trange
from torch.amp import GradScaler, autocast
import math

from utils.utils import DEVICE


scaler = GradScaler()

def compute_gradient_on_subset(model, criterion, dataset_subset, batch_size):
    """
    Compute the average gradient Δ = (1/|D_u|) Σ_{(x,y) in D_u} ∇_θ L(θ, (x,y))
    over the dataset_subset.
    """
    dataloader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False)

    grad_dict = {
        name: torch.zeros_like(param, device=DEVICE)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    total_samples = 0

    model.train()  # ensure gradients are computed
    for inputs, targets in tqdm(dataloader, desc="Computing gradients"):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        model.zero_grad()

        with autocast(device_type=str(DEVICE)):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()

        current_batch = inputs.size(0)
        total_samples += current_batch

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_dict[name] += param.grad.detach().clone() * current_batch

        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         if name not in grad_dict:
        #             grad_dict[name] = param.grad.detach().clone() * current_batch
        #         else:
        #             grad_dict[name] += param.grad.detach().clone() * current_batch

    # Average over total samples
    for name in grad_dict:
        grad_dict[name] /= total_samples

    # Flatten all gradients into one vector
    grad_vector = torch.cat(
        [grad_dict[name].view(-1) for name in sorted(grad_dict.keys())]
    )
    return grad_vector


def lissa_inverse_hvp(
    model, criterion, data_loader, v, damping=1000, scale=1e3, recursion_depth=20
):
    ihvp_estimate = v.clone().to(DEVICE)
    data_iter = iter(data_loader)
    model.train()

    for step in trange(recursion_depth, desc="LiSSA iterations", leave=False):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            inputs, targets = next(data_iter)

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        model.zero_grad()

        # with autocast(device_type=str(DEVICE)):
        #     outputs = model(inputs)
        #     loss = criterion(outputs, targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.nn.utils.parameters_to_vector(grads)

        if torch.isnan(grad_vector).any() or torch.isinf(grad_vector).any():
            print(f"🚨 [Debug] Step {step}: grad_vector has NaN/Inf!")
            return torch.zeros_like(v)

        hv = torch.autograd.grad(
            torch.dot(grad_vector, ihvp_estimate),
            model.parameters(),
            retain_graph=False,
        )
        hv_vector = torch.cat([h.reshape(-1) for h in hv])

        if torch.isnan(hv_vector).any() or torch.isinf(hv_vector).any():
            print(
                f"🚨 [Debug] Step {step}: Hessian-vector product (hv_vector) has NaN/Inf!"
            )
            return torch.zeros_like(v)

        ihvp_estimate = v + (1 - damping) * ihvp_estimate - hv_vector / scale

        if torch.isnan(ihvp_estimate).any() or torch.isinf(ihvp_estimate).any():
            print(f"🚨 [Debug] Step {step}: LiSSA update (ihvp_estimate) has NaN/Inf!")
            return torch.zeros_like(v)

    print("✅ [Debug] LiSSA estimation stable.")
    return ihvp_estimate / scale


def iterative_influence_unlearn(
    model,
    criterion,
    full_dataset,
    removal_indices,
    deletion_batch_size,
    compute_batch_size,
    eps,
    max_norm,
    cg_iters=50,
    scale=1e3,
):
    """
    Implements iterative Influence Unlearning:
      For each mini-batch of deletion samples, compute the average gradient
      Δ_u, solve v ≈ H⁻¹ Δ_u using Conjugate Gradient on the remaining data,
      and update the model as: θ ← θ + v.

    Mathematical Equations:
      Δ_u = (1/|D_u^i|) Σ_{(x,y) in D_u^i} ∇_θ L(θ, (x,y))
      v ≈ H⁻¹ Δ_u,  where H = ∇²_θ L(θ, D \ D_u^i)
      θ ← θ + v
    """
    full_size = len(full_dataset)
    current_indices = set(range(full_size))

    # Partition removal_indices into mini-batches
    removal_list = list(removal_indices)
    num_batches = math.ceil(len(removal_list) / deletion_batch_size)
    partitioned_removals = [
        removal_list[i * deletion_batch_size : (i + 1) * deletion_batch_size]
        for i in range(num_batches)
    ]
    print(
        f"Total deletion samples: {len(removal_list)}; partitioned into {num_batches} mini-batches (each up to {deletion_batch_size} samples)."
    )

    for i, batch in enumerate(tqdm(partitioned_removals, desc="Influence Unlearning")):
        # Update remaining indices: D_current ← D \ D_u^i
        current_indices -= set(batch)
        updated_indices = sorted(list(current_indices))
        dataset_remaining = Subset(full_dataset, updated_indices)
        print(
            f"Iteration {i+1}/{num_batches}: Remaining dataset size = {len(dataset_remaining)}"
        )

        # Compute average gradient Δ_u for the deletion mini-batch
        deleted_subset = Subset(full_dataset, batch)
        delta = compute_gradient_on_subset(
            model, criterion, deleted_subset, compute_batch_size
        )

        if torch.isnan(delta).any() or torch.isinf(delta).any():
            print("🚨 [Debug] Gradient delta contains NaN/Inf!")
        else:
            print("✅ [Debug] Gradient delta stable.")

        # Create a DataLoader for remaining data to approximate Hessian
        remaining_loader = DataLoader(
            dataset_remaining, batch_size=compute_batch_size, shuffle=True
        )

        # # Solve for influence update: v ≈ H⁻¹ Δ_u using Conjugate Gradient
        # influence_update = conjugate_gradient_solver(model, criterion, remaining_loader, delta, cg_iters=cg_iters, damping=eps)

        influence_update = lissa_inverse_hvp(
            model,
            criterion,
            remaining_loader,
            delta,
            damping=eps,
            scale=scale,
            recursion_depth=cg_iters,
        )
        # Optionally clip the update to avoid overly large changes
        update_norm = influence_update.norm(2).item()
        if update_norm > max_norm or math.isnan(update_norm) or math.isinf(update_norm):
            print(
                f"WARNING: Clipping influence update from {update_norm:.2f} to {max_norm}"
            )
            influence_update = influence_update * (max_norm / update_norm)
        print(
            f"Iteration {i+1}: Influence update norm = {influence_update.norm(2).item():.4f}"
        )

        # Update model parameters: θ ← θ + v
        pointer = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    numel = param.numel()
                    update_segment = influence_update[
                        pointer : pointer + numel
                    ].view_as(param)
                    param.data = param.data + update_segment
                    pointer += numel
        print(f"Iteration {i+1}/{num_batches} update completed.")

    return model

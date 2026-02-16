from typing import Optional, Iterable, Sequence, Dict, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

def fisher_unlearn(
    model: nn.Module,
    criterion: nn.Module,
    dataset,
    forget_indices: Optional[Sequence[int]] = None,
    *,
    retain_indices: Optional[Sequence[int]] = None,
    forget_dataset=None,
    retain_dataset=None,
    batch_size: int = 128,
    num_workers: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    fisher_batches: Optional[int] = None,   # limit batch count for Fisher (speed)
    forget_batches: Optional[int] = None,   # limit batch count for forget grad
    alpha: float = 1e-2,                    # step size
    damping: float = 1e-3,                  # (F + damping) in denominator
    only_trainable: bool = True,            # use only params with requires_grad
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Fisher unlearning (diagonal Fisher):
      - compute diag Fisher F on retain data: F = E[g^2]
      - compute mean gradient g_f on forget data
      - update: theta <- theta - alpha * g_f / (F + damping)

    Input options:
      A) provide dataset + forget_indices (+ optional retain_indices)
      B) provide forget_dataset and retain_dataset directly (dataset can be None)

    Returns: (model, stats)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)

    # --------- Build forget/retain datasets ----------
    if forget_dataset is None or retain_dataset is None:
        if dataset is None:
            raise ValueError("Provide either (dataset + forget_indices) or (forget_dataset & retain_dataset).")
        if forget_indices is None:
            raise ValueError("forget_indices must be provided when using full dataset mode.")

        forget_dataset = Subset(dataset, list(forget_indices))

        if retain_indices is None:
            # retain = all other indices
            all_idx = set(range(len(dataset)))
            fset = set(forget_indices)
            retain_indices = sorted(list(all_idx - fset))
        retain_dataset = Subset(dataset, list(retain_indices))

    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=(device.type == "cuda"))
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # --------- Select parameters ----------
    params = []
    param_names = []
    for name, p in model.named_parameters():
        if only_trainable and (not p.requires_grad):
            continue
        params.append(p)
        param_names.append(name)

    if not params:
        raise ValueError("No parameters selected (check requires_grad / only_trainable).")

    # --------- Compute diagonal Fisher on retain ----------
    # Fisher diag approx: E[(dL/dθ)^2] where L is NLL/CE on true labels.
    fisher = [torch.zeros_like(p, device=device) for p in params]
    total_ret = 0

    model.eval()  # important: disable dropout/bn randomness
    for b, batch in enumerate(retain_loader):
        if fisher_batches is not None and b >= fisher_batches:
            break

        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        bs = x.shape[0]
        total_ret += bs

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)

        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=False)

        for i, g in enumerate(grads):
            fisher[i] += (g.detach() ** 2) * bs  # weight by batch size

    if total_ret == 0:
        raise ValueError("Retain loader produced 0 samples.")

    for i in range(len(fisher)):
        fisher[i] /= float(total_ret)
        fisher[i] += damping  # stabilize denominator

    # --------- Compute mean gradient on forget ----------
    g_forget = [torch.zeros_like(p, device=device) for p in params]
    total_forget = 0

    model.eval()
    for b, batch in enumerate(forget_loader):
        if forget_batches is not None and b >= forget_batches:
            break

        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        bs = x.shape[0]
        total_forget += bs

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)

        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=False)

        for i, g in enumerate(grads):
            g_forget[i] += g.detach() * bs

    if total_forget == 0:
        raise ValueError("Forget loader produced 0 samples.")

    for i in range(len(g_forget)):
        g_forget[i] /= float(total_forget)

    # --------- Apply Fisher-preconditioned update ----------
    model.train()  # back to train mode for further training if you want

    with torch.no_grad():
        for i, p in enumerate(params):
            p -= alpha * (g_forget[i] / fisher[i])


    return model
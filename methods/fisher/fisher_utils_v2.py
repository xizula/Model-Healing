import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from typing import Callable, Dict, List, Any, Optional, Tuple

def _default_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device

def _extract_xy(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        for kx, ky in (("x", "y"), ("inputs", "targets"), ("input", "target")):
            if kx in batch and ky in batch:
                return batch[kx], batch[ky]
    raise TypeError("Unsupported batch format. Expected (x,y) or dict with x/y keys.")

def empirical_fisher_diag_streaming(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    eps: float = 1e-8,
    max_samples: Optional[int] = None,
    microbatch_size: int = 1,
    use_amp: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Empirical Fisher diagonal:
      F_diag ≈ E_i[ (∂ loss_i / ∂θ)^2 ]

    Memory-safe streaming implementation:
      - Computes per-sample gradients in microbatches (default 1).
      - Accumulates sum of squares into fisher tensors.
    """
    if device is None:
        device = _default_device(model)

    model.eval()

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    fisher = {n: torch.zeros_like(p, device=device) for n, p in params.items()}

    seen = 0
    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else torch.cpu.amp.autocast

    # Important: for per-sample grads we must avoid reduction across batch.
    # We'll call loss_fn on a microbatch and then sum individual losses by looping samples,
    # OR enforce reduction='none'. We do the robust method: compute per-sample loss if possible.
    for batch in dataloader:
        x, y = _extract_xy(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        bsz = x.size(0)
        if max_samples is not None and seen >= max_samples:
            break
        if max_samples is not None and seen + bsz > max_samples:
            bsz = max_samples - seen
            x = x[:bsz]
            y = y[:bsz]

        # Split into microbatches
        for start in range(0, bsz, microbatch_size):
            end = min(start + microbatch_size, bsz)
            xm = x[start:end]
            ym = y[start:end]

            # We need per-sample grads => easiest is loop samples in microbatch.
            # microbatch_size=1 is safest memory-wise.
            for i in range(xm.size(0)):
                model.zero_grad(set_to_none=True)
                xi = xm[i : i + 1]
                yi = ym[i : i + 1]

                with amp_ctx():
                    out = model(xi)
                    li = loss_fn(out, yi)
                    # ensure scalar
                    if li.ndim != 0:
                        li = li.mean()

                li.backward()

                with torch.no_grad():
                    for name, p in params.items():
                        if p.grad is not None:
                            fisher[name].add_(p.grad.detach() ** 2)

                seen += 1
                if max_samples is not None and seen >= max_samples:
                    break

            if max_samples is not None and seen >= max_samples:
                break

        if max_samples is not None and seen >= max_samples:
            break

    # Average and clamp
    denom = max(seen, 1)
    for name in fisher:
        fisher[name].div_(denom)
        fisher[name].clamp_min_(eps)

    return fisher


@torch.no_grad()
def fisher_forgetting_apply_noise(
    model: nn.Module,
    fisher_diag: Dict[str, torch.Tensor],
    *,
    alpha: float,
    exponent: float = -0.25,
    eps: float = 1e-8,
    generator: Optional[torch.Generator] = None,
) -> None:
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scale = (fisher_diag[name] + eps).pow(exponent)
        noise = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=generator)
        p.add_(alpha * scale * noise)


def fisher_forgetting_unlearn(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    full_dataset,
    removal_entries: List[dict],
    fisher_batch_size: int = 32,      # batching for loading data (not per-sample grad)
    microbatch_size: int = 1,         # per-sample grad microbatch, keep 1-4 on 6GB GPU
    alpha: float = 0.05,
    exponent: float = -0.25,
    eps: float = 1e-8,
    device: Optional[torch.device] = None,
    max_fisher_samples: Optional[int] = 50_000,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: Optional[int] = None,
    use_amp: bool = True,
) -> nn.Module:
    if device is None:
        device = _default_device(model)
    model.to(device)

    removal_set = set(removal_entries)
    remaining_indices = [i for i in range(len(full_dataset)) if i not in removal_set]
    remaining_subset = Subset(full_dataset, remaining_indices)

    loader = DataLoader(
        remaining_subset,
        batch_size=fisher_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Optional: reduce fragmentation / cached memory behavior
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fisher_diag = empirical_fisher_diag_streaming(
        model,
        loss_fn,
        loader,
        device=device,
        eps=eps,
        max_samples=max_fisher_samples,
        microbatch_size=microbatch_size,
        use_amp=use_amp,
    )

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    fisher_forgetting_apply_noise(
        model,
        fisher_diag,
        alpha=alpha,
        exponent=exponent,
        eps=eps,
        generator=gen,
    )

    return model

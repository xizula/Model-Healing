import torch
from torch.utils.data import DataLoader, Subset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_diag_fisher(model, dataset_subset, batch_size, criterion):
    model.eval()
    loader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False)

    fisher = {}
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        bs = x.size(0)
        total += bs

        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            g2 = (p.grad.detach() ** 2) * bs
            fisher[name] = g2 if name not in fisher else (fisher[name] + g2)

    for name in fisher:
        fisher[name] /= max(total, 1)

    return fisher


@torch.no_grad()
def add_fisher_noise_stable_(
    model,
    fisher_diag,
    noise_scale,
    damping=1e-3,            # KLUCZOWE: zamiast 1e-8
    fisher_power=-0.25,      # -0.25 zgodnie z uproszczeniem; możesz testować -0.5 :contentReference[oaicite:5]{index=5}
    max_rel_rms=0.05,        # max RMS szumu jako % RMS wag (0.02–0.10 typowo)
    skip_bias_bn=True,       # biasy i BN często lepiej oszczędzić
):
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        F = fisher_diag[name].to(p.device)

        # (F + damping)^power zamiast (F + eps)^power
        scale = (F + damping).pow(fisher_power)

        noise = torch.randn_like(p)
        delta = noise_scale * scale * noise

        # Clip względnej wielkości szumu per tensor (żeby uniknąć collapse)
        w_rms = p.pow(2).mean().sqrt().clamp_min(1e-12)
        d_rms = delta.pow(2).mean().sqrt()

        max_allowed = max_rel_rms * w_rms
        if d_rms > max_allowed:
            delta = delta * (max_allowed / d_rms)

        p.add_(delta)


def fisher_unlearn(
    model,
    criterion,
    full_dataset,
    removal_indices,
    batch_size=64,
    lambda_=5e-7,     # jak w paper :contentReference[oaicite:6]{index=6}
    sigma_h=1.0,
    damping=1e-3,
    fisher_power=-0.25,
    max_rel_rms=0.05,
):
    model = model.to(DEVICE)

    removal_set = set(removal_indices)
    retain_indices = [i for i in range(len(full_dataset)) if i not in removal_set]
    Dr = Subset(full_dataset, retain_indices)

    fisher_diag = compute_diag_fisher(model, Dr, batch_size=batch_size, criterion=criterion)

    # skala z paper: (lambda * sigma_h^2)^(1/4) :contentReference[oaicite:7]{index=7}
    noise_scale = (lambda_ * (sigma_h ** 2)) ** 0.25

    model.eval()
    add_fisher_noise_stable_(
        model,
        fisher_diag,
        noise_scale=noise_scale,
        damping=damping,
        fisher_power=fisher_power,
        max_rel_rms=max_rel_rms,
        skip_bias_bn=True,
    )

    return model

import math
import torch
from torch.utils.data import DataLoader, Subset

# Jeśli masz u siebie stałą DEVICE, to podmień:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _add_fisher_noise_(model, fisher_diag, noise_scale, eps=1e-8, fisher_power=-0.25):
    """
    Wstrzykuje szum: w <- w + noise_scale * (F + eps)^(fisher_power) * N(0, I)
    fisher_power = -0.25 odpowiada F^{-1/4} z artykułu,
    fisher_power = -0.5  odpowiada F^{-1/2} (wariant praktyczny z uwag implementacyjnych).
    """
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        F = fisher_diag[name].to(p.device)
        scale = (F + eps).pow(fisher_power)
        noise = torch.randn_like(p)
        p.add_(noise_scale * scale * noise)


def compute_diag_fisher(
    model,
    dataset_subset,
    batch_size,
    criterion=None,
    use_true_labels=True,
):
    """
    Diagonal Fisher.

    - Jeśli use_true_labels=True: empirical fisher na podstawie grad(loss) dla (x, y_true)
      (to jest to, co robi Wasz kod).
    - Jeśli use_true_labels=False: bliżej definicji FIM: próbkujemy y ~ p(y|x) i liczymy grad log p(y|x)
      (zgodniejsze z eq. (8) w artykule).
    """
    model.eval()
    loader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False)

    fisher = {}
    total = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        model.zero_grad(set_to_none=True)

        logits = model(x)

        if use_true_labels:
            if criterion is None:
                raise ValueError("criterion wymagany, gdy use_true_labels=True")
            loss = criterion(logits, y)
        else:
            # log p(y|x) z y~p(y|x)
            probs = torch.softmax(logits, dim=1)
            y_sample = torch.multinomial(probs, num_samples=1).squeeze(1)
            logp = torch.log(torch.gather(probs, 1, y_sample.unsqueeze(1)).squeeze(1) + 1e-12)
            loss = (-logp).mean()

        loss.backward()

        bs = x.size(0)
        total += bs

        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            g2 = (p.grad.detach() ** 2) * bs
            fisher[name] = g2 if name not in fisher else (fisher[name] + g2)

    for name in fisher:
        fisher[name] = fisher[name] / max(total, 1)

    return fisher


def fisher_unlearn(
    model,
    criterion,
    full_dataset,
    removal_indices,
    batch_size=64,
    lambda_=5e-7,
    sigma_h=1.0,
    eps=1e-8,
    fisher_power=-0.25,
    use_true_labels=True,
):
    """
    Fisher forgetting (zgodnie z artykułem – wariant DNN):
      1) Dr = D \\ Df
      2) policz diag(F) na Dr
      3) S(w) = w + (lambda * sigma_h^2)^(1/4) * F^(power) * epsilon, gdzie power=-1/4

    Parametry:
      - lambda_: trade-off (większe => więcej szumu => mocniejsze zapominanie, gorsza dokładność)
      - sigma_h: parametr błędu aproksymacji (w praktyce często ustawiany „na oko”)
      - fisher_power: -0.25 (F^{-1/4}) jak w tekście; opcjonalnie -0.5 (F^{-1/2})
    """
    model = model.to(DEVICE)

    full_size = len(full_dataset)
    removal_set = set(removal_indices)
    retain_indices = [i for i in range(full_size) if i not in removal_set]
    dataset_retain = Subset(full_dataset, retain_indices)

    fisher_diag = compute_diag_fisher(
        model=model,
        dataset_subset=dataset_retain,
        batch_size=batch_size,
        criterion=criterion,
        use_true_labels=use_true_labels,
    )

    noise_scale = (lambda_ * (sigma_h ** 2)) ** 0.25
    with torch.no_grad():
        _add_fisher_noise_(
            model=model,
            fisher_diag=fisher_diag,
            noise_scale=noise_scale,
            eps=eps,
            fisher_power=fisher_power,
        )

    return model

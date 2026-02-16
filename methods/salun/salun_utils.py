import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn


# =========================
# Helpers (zamiast utils.py)
# =========================

class AverageMeter:
    """Trzyma bieżącą wartość, średnią itd. (jak w repo)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy dla klasyfikacji."""
    # logits: (B, C), target: (B,)
    pred = logits.argmax(dim=1)
    correct = (pred == target).float().sum().item()
    return 100.0 * correct / max(1, target.numel())


def warmup_lr(
    epoch: int,
    step_in_epoch: int,
    optimizer: torch.optim.Optimizer,
    *,
    one_epoch_steps: int,
    base_lr: float,
    warmup_epochs: int,
):
    """
    Linear warmup LR (jak w repo, tylko jasno i bez zależności od args).
    - epoch: aktualny epoch (0-index albo 1-index nie ma znaczenia, byle spójnie)
    - step_in_epoch: krok w epoce (1..one_epoch_steps)
    """
    if warmup_epochs <= 0:
        return
    if epoch >= warmup_epochs:
        return

    # progress w [0, 1]
    total_warmup_steps = warmup_epochs * one_epoch_steps
    current_step = epoch * one_epoch_steps + step_in_epoch
    lr = base_lr * float(current_step) / float(max(1, total_warmup_steps))

    for pg in optimizer.param_groups:
        pg["lr"] = lr


# =========================
# RL Unlearning (1 plik)
# =========================

@dataclass
class RLHistory:
    retain_loss_avg: float
    retain_acc_avg: float
    steps_forget: int
    steps_retain: int


def salun_unlearn_rl(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    forget_loader: torch.utils.data.DataLoader,
    retain_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int = 0,
    num_classes: int = 10,
    device: Optional[torch.device | str] = None,
    print_freq: int = 50,
    warmup_epochs: int = 0,
    base_lr: Optional[float] = None,
    mask: Optional[Dict[str, torch.Tensor]] = None,
    use_amp: bool = False,
) -> Tuple[float, RLHistory]:
    """
    RL (Random Labels) unlearning:
      1) trenuj na forget_loader z losowymi etykietami
      2) trenuj na retain_loader z prawdziwymi etykietami + licz metryki na retain

    Zwraca:
      - retain_top1_avg (float) oraz RLHistory

    Parametry:
      - num_classes: MNIST=10, CIFAR10=10, AnimalFaces = liczba klas
      - warmup_epochs: ustaw 0 żeby wyłączyć
      - base_lr: jeśli None, bierze z optimizer.param_groups[0]['lr']
      - mask: dict {param_name: tensor_mask} do mnożenia gradientów (opcjonalnie)
      - use_amp: True jeśli chcesz AMP na CUDA
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)
    model.train()

    if base_lr is None:
        base_lr = float(optimizer.param_groups[0].get("lr", 1e-3))

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    losses = AverageMeter()
    top1 = AverageMeter()

    loader_len = len(forget_loader) + len(retain_loader)
    t0 = time.time()

    # -----------------------
    # (1) FORGET: random label
    # -----------------------
    for i, batch in enumerate(tqdm(forget_loader, desc="Forget loader processing...")):
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)

        # losowe etykiety o tym samym kształcie co y
        y_shape = torch.as_tensor(y).shape
        y_rand = torch.randint(
            low=0, high=num_classes, size=y_shape,
            device=device, dtype=torch.long
        )

        # warmup (kroki liczone globalnie w epoce)
        warmup_lr(
            epoch=epoch,
            step_in_epoch=i + 1,
            optimizer=optimizer,
            one_epoch_steps=loader_len,
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y_rand)

        scaler.scale(loss).backward()

        if mask is not None:
            scaler.unscale_(optimizer)
            for name, p in model.named_parameters():
                if p.grad is not None:
                    p.grad.mul_(mask[name])

        scaler.step(optimizer)
        scaler.update()

    # -----------------------
    # (2) RETAIN: true labels + metrics
    # -----------------------
    for j, batch in enumerate(tqdm(retain_loader, desc="Retain loader processing...")):
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device, dtype=torch.long)

        global_step = len(forget_loader) + j
        warmup_lr(
            epoch=epoch,
            step_in_epoch=global_step + 1,
            optimizer=optimizer,
            one_epoch_steps=loader_len,
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if mask is not None:
            scaler.unscale_(optimizer)
            for name, p in model.named_parameters():
                if p.grad is not None:
                    p.grad.mul_(mask[name])

        scaler.step(optimizer)
        scaler.update()

        acc1 = accuracy_top1(logits.detach(), y)
        losses.update(float(loss.detach().cpu()), x.size(0))
        top1.update(float(acc1), x.size(0))

        if print_freq > 0 and (j + 1) % print_freq == 0:
            t1 = time.time()
            print(
                f"Epoch [{epoch}] Retain [{j+1}/{len(retain_loader)}]  "
                f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                f"Acc@1 {top1.val:.2f}% ({top1.avg:.2f}%)  "
                f"Time {t1 - t0:.2f}s"
            )
            t0 = time.time()

    hist = RLHistory(
        retain_loss_avg=float(losses.avg),
        retain_acc_avg=float(top1.avg),
        steps_forget=len(forget_loader),
        steps_retain=len(retain_loader),
    )
    return model, hist

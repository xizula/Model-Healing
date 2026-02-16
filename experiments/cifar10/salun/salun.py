from pathlib import Path
import sys
import copy
from torchvision import datasets
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm, trange
import numpy as np
import time

project_root = Path.cwd().resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

data_root = project_root / "data"
data_root.mkdir(parents=True, exist_ok=True)

from utils.utils import DEVICE

print(f"Device used: {DEVICE}")

# Set random seed for reproducibility
from utils.utils import set_seed

set_seed()

from utils.utils import save_model

from models.resnet50 import load_model_resnet50, init_model_resnet50

# Merics
from utils.train_test_metrics import test_model, show_metrics

# Recreate Dataloaders from json files
from methods.naive.naive_utils import recreate_dataloaders

# # Fisher Information Matrix (FIM) calc and unlearning with FIM
from methods.fisher.fisher_utils import create_unlearning_dataloader, create_retain_dataloader
from methods.influence.influence_utils import iterative_influence_unlearn
from methods.salun.salun_utils import salun_unlearn_rl

BATCH_SIZE = 64
MINI_BATCH_SIZE = 1024

naive_results_path = Path("D:/Unlearning/results/cifar10/naive")
results_path = Path("D:/Unlearning/results/cifar10/salun")

model_file = naive_results_path / "ResNet50_CIFAR10_model.pth"
samples_to_unlearn_file = (
    naive_results_path / "cifar10_samples_to_unlearn_30per.json"
)
remaining_dataset_file = (
    naive_results_path / "updated_cifar10_data_splits.json"
)

original_model, original_model_name, criterion, _optimizer, transform = (
    load_model_resnet50(model_pth_path=model_file)
)

model_to_unlearn = copy.deepcopy(original_model)
model_to_unlearn_name = "salun_" + original_model_name

train_dataset = datasets.CIFAR10(
    root=data_root, train=True, transform=transform, download=True
)
test_dataset = datasets.CIFAR10(
    root=data_root, train=False, transform=transform, download=True
)

unlearn_indices, unlearn_loader = create_unlearning_dataloader(
    samples_to_unlearn_file, train_dataset, batch_size=BATCH_SIZE
)

retain_indices, retain_loader = create_retain_dataloader(
    samples_to_unlearn_file, train_dataset, batch_size=BATCH_SIZE
)


optimizer = torch.optim.SGD(model_to_unlearn.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

unlearned_model, hist = salun_unlearn_rl(
    model = model_to_unlearn,
    criterion = criterion,
    forget_loader = unlearn_loader,
    retain_loader = retain_loader,
    optimizer = optimizer,
    epoch = 0,
    print_freq = 50,
    warmup_epochs = 0,
    base_lr = None,
    num_classes = 10,
    mask = None,
    device = DEVICE,
    use_amp = False,
)


save_model(unlearned_model, results_path / model_to_unlearn_name + "_model.pth")

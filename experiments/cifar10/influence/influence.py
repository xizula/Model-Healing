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
from methods.fisher.fisher_utils import create_unlearning_dataloader
from methods.influence.influence_utils import iterative_influence_unlearn

BATCH_SIZE = 64
MINI_BATCH_SIZE = 1024

EPS = 1000
MAX_NORM = 10  # 0.5
CG_ITERS = 5
SCALE = 1e3

naive_results_path = Path("D:/Unlearning/results/cifar10/naive")
results_path = Path("D:/Unlearning/results/cifar10/influence")

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
model_to_unlearn_name = "influence_" + original_model_name

train_dataset = datasets.CIFAR10(
    root=data_root, train=True, transform=transform, download=True
)
test_dataset = datasets.CIFAR10(
    root=data_root, train=False, transform=transform, download=True
)

unlearn_indices, unlearn_loader = create_unlearning_dataloader(
    samples_to_unlearn_file, train_dataset, batch_size=MINI_BATCH_SIZE
)

start_time = time.perf_counter()


    # Your main influence unlearning script goes here.
unlearned_model = iterative_influence_unlearn(
    model=model_to_unlearn,
    criterion=criterion,
    full_dataset=train_dataset,
    removal_indices=unlearn_indices,
    deletion_batch_size=MINI_BATCH_SIZE,
    compute_batch_size=BATCH_SIZE,
    eps=EPS,
    max_norm=MAX_NORM,
    cg_iters=CG_ITERS,
    scale=SCALE,
)

end_time = time.perf_counter()  # End timer
elapsed_time = end_time - start_time

save_model(unlearned_model, results_path / f"{model_to_unlearn_name}_model.pth")

print(f"Execution time: {elapsed_time:.6f} seconds")
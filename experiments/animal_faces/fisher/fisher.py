from pathlib import Path
import sys
import copy
from torchvision import datasets
from tqdm.notebook import tqdm


project_root = Path.cwd().resolve()
print(project_root)
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

from models.effnetb0 import load_model_effnetb0, init_model_effnetb0

# Merics
from utils.train_test_metrics import test_model, show_metrics

# Recreate Dataloaders from json files
from methods.naive.naive_utils import recreate_dataloaders

# Fisher Information Matrix (FIM) calc and unlearning with FIM
from methods.fisher.fisher_utils import (
    iterative_fisher_unlearn,
    create_unlearning_dataloader,
)

from methods.fisher.fisher_utils_v2 import fisher_forgetting_unlearn
from methods.fisher.fisher_utils_v3 import fisher_unlearn


BATCH_SIZE = 32
MINI_BATCH_SIZE = 1

SIGMA = 1

EPS = 1e-3
MAX_NORM = 0

import math
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from utils.utils import DEVICE



model_file = project_root / "results/animal_faces/naive/EffNetB0_AFHQ_model.pth"
samples_to_unlearn_file = (
    project_root / "results/animal_faces/naive/afhq_samples_to_unlearn_30per.json"
)
remaining_dataset_file = (
    project_root / "results/animal_faces/naive/updated_afhq_data_splits.json"
)

original_model, original_model_name, criterion, _optimizer, transform = (
    load_model_effnetb0(model_pth_path=model_file)
)

model_to_unlearn = copy.deepcopy(original_model)
import torch.nn as nn

# model_to_unlearn = nn.DataParallel(model_to_unlearn, device_ids=[0, 1])

model_to_unlearn_name = "fisher_" + original_model_name

train_dataset = datasets.ImageFolder(root=data_root / "afhq/train", transform=transform)
test_dataset = datasets.ImageFolder(root=data_root / "afhq/val", transform=transform)

unlearn_indices, _unlearn_loader = create_unlearning_dataloader(
    samples_to_unlearn_file, train_dataset, batch_size=MINI_BATCH_SIZE
)

import time

start_time = time.perf_counter()

unlearned_model = iterative_fisher_unlearn(
    model_to_unlearn,
    criterion,
    train_dataset,
    unlearn_indices,
    SIGMA,
    deletion_batch_size=MINI_BATCH_SIZE,
    compute_batch_size=BATCH_SIZE,
    eps=EPS,
    max_norm=MAX_NORM,
)

# unlearned_model = fisher_forgetting_unlearn(
#     model=model_to_unlearn,
#     loss_fn=criterion,
#     fisher_batch_size=BATCH_SIZE,
#     microbatch_size=MINI_BATCH_SIZE,
#     full_dataset=train_dataset,
#     removal_entries=unlearn_indices,
#     alpha=SIGMA,
#     eps=EPS,
#     device=DEVICE,
#     max_fisher_samples=10000
# )

# unlearned_model = fisher_unlearn(
#     model_to_unlearn,
#     criterion,
#     train_dataset,
#     unlearn_indices
#     )

end_time = time.perf_counter()  # End timer
elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time:.6f} seconds")
save_model(unlearned_model, project_root / "results/animal_faces/fisher" / f"{model_to_unlearn_name}_model.pth")
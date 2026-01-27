import sys
import json
from pathlib import Path
from torchvision import datasets
import time

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

# Init model
from models.resnet50 import init_model_resnet50

# Create sisa structure
from methods.sisa.sisa_utils import create_sisa_structure

# Recreate sisa dataloaders from json file
from methods.sisa.sisa_utils import recreate_sisa_dataloaders

# SISA train & test loop
import methods.sisa.sisa_train_test as stt
from methods.sisa.sisa_train_test import sisa_train, sisa_test, retrain_sisa_framework

# Aggregate SISA models, via weighted voting
from methods.sisa.sisa_utils import evaluate_aggregated_model, update_sisa_structure

BATCH_SIZE = 64
LEARNING_RATE = 5e-5
EPOCHS = 5
SHARDS = 3
SLICES = 5

FIRST_TRAIN = False


*_, transform = init_model_resnet50()
train_dataset = datasets.CIFAR10(
    root=data_root, train=True, transform=transform, download=True
)
test_dataset = datasets.CIFAR10(
    root=data_root, train=False, transform=transform, download=True
)
results_path = Path("D:/Unlearning/results/cifar10/sisa")
sisa_structure_file = results_path / "sisa_structure.json"
unlearn_samples_file = Path("D:/Unlearning/results/cifar10/naive") / "cifar10_samples_to_unlearn_30per.json"
sisa_structure = results_path / "sisa_structure.json"
updated_sisa_structure = results_path / "updated_sisa_strucute.json"
deleted_samples = results_path / "deleted_samples.json"

if FIRST_TRAIN:
    create_sisa_structure(train_dataset, shards=SHARDS, slices_per_shard=SLICES, results_path=sisa_structure_file)
    dataloaders, classes = recreate_sisa_dataloaders(
        datasets=(train_dataset, test_dataset),
        info_file_path=sisa_structure_file,
        batch_size=BATCH_SIZE,
        val_ratio=0.1,
    )

    save_models_metrics_dir = results_path / "sisa_models"
    start_time = time.perf_counter()
    sisa_train(
        dataloaders=dataloaders,
        num_epochs=EPOCHS,
        save_models_metrics_dir=save_models_metrics_dir,
        init_model_func=init_model_resnet50,
        learning_rate=LEARNING_RATE,
    )

    end_time = time.perf_counter()  # End timer
    elapsed_time = end_time - start_time

    print(f"Execution time: {elapsed_time:.6f} seconds")

else:
    affected_shards = update_sisa_structure(
        unlearn_samples_file, sisa_structure, updated_sisa_structure, deleted_samples
    )

    dataloaders, classes = recreate_sisa_dataloaders(
        datasets=(train_dataset, test_dataset),
        info_file_path=updated_sisa_structure,
        batch_size=BATCH_SIZE,
        val_ratio=0.1,
    )

    save_path = results_path / "sisa_updated_models"

    start_time = time.perf_counter()

    retrain_sisa_framework(
        dataloaders=dataloaders,
        affected_shards=affected_shards,
        num_epochs=EPOCHS,
        save_models_metrics_dir=save_path,
        init_model_func=init_model_resnet50,
        learning_rate=LEARNING_RATE,
    )

    end_time = time.perf_counter()  # End timer
    elapsed_time = end_time - start_time

    print(f"Execution time: {elapsed_time:.6f} seconds")
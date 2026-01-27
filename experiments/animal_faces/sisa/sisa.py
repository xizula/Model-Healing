from pathlib import Path
import sys

project_root = Path.cwd().resolve()
print(project_root)
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

data_root = project_root / "data"
data_root.mkdir(parents=True, exist_ok=True)

from models.effnetb0 import init_model_effnetb0

# Create sisa structure
from methods.sisa.sisa_utils import create_sisa_structure

# Recreate sisa dataloaders from json file
from methods.sisa.sisa_utils import recreate_sisa_dataloaders

# SISA train & test loop
from methods.sisa.sisa_train_test import sisa_train, sisa_test, retrain_sisa_framework

# Aggregate SISA models, via weighted voting
from methods.sisa.sisa_utils import evaluate_aggregated_model, update_sisa_structure
from torchvision import datasets
from utils.utils import DEVICE
import time

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 3
SHARDS = 3
SLICES = 5

*_, transform = init_model_effnetb0()

# SET TRUE FOR RETRAINING - UNLEARNING
FIRST_TRAIN = False

train_dataset = datasets.ImageFolder(root=data_root / "afhq/train", transform=transform)
test_dataset = datasets.ImageFolder(root=data_root / "afhq/val", transform=transform)
results_path = Path("D:/Unlearning/results/animal_faces/sisa")

if FIRST_TRAIN:
    sisa_structure_file = results_path / "sisa_structure.json"
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
        init_model_func=init_model_effnetb0,
        learning_rate=LEARNING_RATE,
    )

    end_time = time.perf_counter()  # End timer
    elapsed_time = end_time - start_time

else:
    unlearn_samples_file = Path("D:/Unlearning/results/animal_faces/naive") / "afhq_samples_to_unlearn_30per.json"
    sisa_structure = results_path / "sisa_structure.json"
    updated_sisa_structure = results_path / "updated_sisa_strucute.json"
    deleted_samples = results_path / "deleted_samples.json"

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
        init_model_func=init_model_effnetb0,
        learning_rate=LEARNING_RATE,
    )

    end_time = time.perf_counter()  # End timer
    elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time:.6f} seconds")
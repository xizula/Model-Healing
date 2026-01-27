# Select samples to unlearn (10% random)
import sys
from pathlib import Path
import sys
import time

project_root = Path.cwd().resolve()
print(project_root)
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

data_root = project_root / "data"
data_root.mkdir(parents=True, exist_ok=True)

from utils.utils import select_samples_to_unlearn
from models.effnetb0 import init_model_effnetb0
from methods.naive.naive_utils import init_dataloaders
# Update data splits
from methods.naive.naive_utils import update_splits_after_unlearning

# Recreate Dataloaders from json files
from methods.naive.naive_utils import recreate_dataloaders
from utils.train_test_metrics import train_model
from torchvision import datasets
from utils.utils import DEVICE

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 3
FIRST_TRAIN = False

results_path = Path("D:/Unlearning/results/animal_faces/naive")
data_splits_file = results_path / "afhq_data_splits.json"
unlearn_samples_file = results_path / "afhq_samples_to_unlearn_30per.json"
updated_data_splits_path = results_path / "updated_afhq_data_splits.json"

model, model_name, criterion, optimizer, transform = init_model_effnetb0(
    learning_rate=LEARNING_RATE, fc_output=3
)

train_dataset = datasets.ImageFolder(root=data_root / "afhq/train", transform=transform)
test_dataset = datasets.ImageFolder(root=data_root / "afhq/val", transform=transform)

if FIRST_TRAIN:
    data_split_path = "afhq_data_splits.json"

    train_loader, val_loader, test_loader, classes = init_dataloaders(
        datasets=(train_dataset, test_dataset),
        val_ratio=0.2,
        batch_size=BATCH_SIZE,
        info_file_path=data_splits_file,
    )
    start_time = time.perf_counter()
    train_model(
        model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, results_path=results_path
    )
    end_time = time.perf_counter()  # End timer
    elapsed_time = end_time - start_time

else:
    model_name = "naive_unlearning_" + model_name
    select_samples_to_unlearn(data_splits_file, unlearn_samples_file, unlearn_ratio=0.3)

    update_splits_after_unlearning(
        data_splits_file, unlearn_samples_file, updated_data_splits_path
    )

    train_loader, val_loader, test_loader, classes = recreate_dataloaders(
        data_splits_file=updated_data_splits_path,
        datasets=(train_dataset, test_dataset),
        batch_size=BATCH_SIZE,
    )

    import time

    start_time = time.perf_counter()
    train_model(
        model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, results_path=results_path
    )
    end_time = time.perf_counter()  # End timer
    elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time:.6f} seconds")
import sys
from pathlib import Path

project_root = Path.cwd().resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

data_root = project_root / "data"

from torchvision import datasets
from utils.utils import DEVICE
import sys


print(f"Device used: {DEVICE}")

# Set random seed for reproducibility
from utils.utils import set_seed
import time
set_seed()

# Prepare Dataloaders
from methods.naive.naive_utils import init_dataloaders

# Train loop
from utils.train_test_metrics import train_model

# Plot losses
from utils.train_test_metrics import plot_training_history

# Test function
from utils.train_test_metrics import test_model

# Merics
from utils.train_test_metrics import show_metrics

# Init model
from models.effnetb0 import init_model_effnetb0

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 3


model, model_name, criterion, optimizer, transform = init_model_effnetb0(
    learning_rate=LEARNING_RATE, fc_output=3
)

train_dataset = datasets.ImageFolder(root=data_root / "afhq/train", transform=transform)
test_dataset = datasets.ImageFolder(root=data_root / "afhq/val", transform=transform)
results_path = project_root / "results/animal_faces/naive"
data_split_path = results_path / "afhq_data_splits.json"



train_loader, val_loader, test_loader, classes = init_dataloaders(
    datasets=(train_dataset, test_dataset),
    val_ratio=0.2,
    batch_size=BATCH_SIZE,
    info_file_path=data_split_path,
)

start_time = time.perf_counter()
train_model(
    model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, results_path=results_path,
)
end_time = time.perf_counter()  # End timer
elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time:.6f} seconds")
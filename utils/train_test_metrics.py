import json
import torch
import numpy as np
from tqdm import tqdm
import seaborn as sb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from matplotlib import pyplot as plt

from utils.utils import save_model, DEVICE


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=10,
    influence_sigma=None,
    scheduler=None,
    results_path=".",
):

    best_val_accuracy = 0.0
    best_model_path = results_path / f"{model_name}_model.pth"

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0

        all_preds = []
        all_labels = []

        # Training phase
        for inputs, labels in tqdm(
            train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"
        ):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # Influence noise
            if influence_sigma is not None:
                theta_vec = torch.cat([p.view(-1) for p in model.parameters()])
                b = torch.randn_like(
                    theta_vec, device=DEVICE
                )  # Sample b from a Normal(0, I) distribution
                dot_bt_theta = torch.dot(b, theta_vec)  # Dot product b^T theta
                noise_term = (
                    influence_sigma / len(train_loader.dataset)
                ) * dot_bt_theta  # Scale by (sigma / |D|)
                loss += noise_term

            # Backward pass and optimize
            loss.backward()

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = accuracy_score(all_labels, all_preds)

        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_accuracy)

        # Validation phase
        model.eval()

        val_running_loss = 0.0

        val_preds = []
        val_labels = []

        with torch.inference_mode():

            for inputs, labels in tqdm(
                val_loader, desc=f"Evaluating on validation set..."
            ):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = accuracy_score(val_labels, val_preds)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Print training and validation results
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, best_model_path)
            print(
                f"Epoch {epoch + 1}: New best validation accuracy: {best_val_accuracy:.4f}. Model saved to {best_model_path}."
            )

        if scheduler is not None:
            scheduler.step()

    with open(results_path / f"{model_name}_history.json", "w") as f:
        json.dump(history, f)

    print(
        f"Training complete for {model_name}. Training stats saved to '{model_name}_history.json'."
    )


def test_model(model, model_name, model_path, test_loader, results_path):

    print(f"Loading and testing model: {model_name}")
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=DEVICE)
    )

    model.eval()

    predictions = []
    true_labels = []

    with torch.inference_mode():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating model: {model_path}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())

    results = {"predictions": [predictions], "true_labels": [true_labels]}

    with open(results_path / f"{model_name}_predictions.json", "w") as f:
        json.dump(results, f)

    print(f"Predictions and labels saved to {model_name}_predictions.json")


def plot_training_history(history_path):

    with open(history_path, "r") as f:
        data = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(data["train_loss"], label="Train Loss")
    plt.plot(data["val_loss"], label="Val Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(data["train_accuracy"], label="Train Accuracy")
    plt.plot(data["val_accuracy"], label="Val Accuracy", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def show_metrics(predictions_path, classes, model_name):

    with open(predictions_path, "r") as f:
        data = json.load(f)

    predictions = np.array(data["predictions"]).flatten().tolist()
    true_labels = np.array(data["true_labels"]).flatten().tolist()

    # Metrics calculation
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(
        true_labels, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
    cm = confusion_matrix(true_labels, predictions)

    # Display metrics
    print(f"Metrics for {model_name}:")
    print(f"  - Test Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1 Score: {f1:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sb.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

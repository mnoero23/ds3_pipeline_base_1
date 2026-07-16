"""Pipeline base de entrenamiento y validación en PyTorch sobre Iris."""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
OUTPUT_DIR = Path("results")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders() -> tuple[DataLoader, DataLoader, int, int]:
    iris = load_iris()
    X_train, X_val, y_train, y_val = train_test_split(
        iris.data,
        iris.target,
        test_size=0.20,
        random_state=SEED,
        stratify=iris.target,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, iris.data.shape[1], len(iris.target_names)


class IrisMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_artifacts(history: dict[str, list[float]], metadata: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"metadata": metadata, "history": history}, file, indent=2)

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="Loss por época", xlabel="Época", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy por época", xlabel="Época", ylabel="Accuracy", ylim=(0, 1.05))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(figure)


def main() -> None:
    set_seed()
    device = get_device()
    train_loader, val_loader, input_dim, num_classes = build_loaders()
    model = IrisMLP(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {key: [] for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy")}

    print(f"Device: {device} | PyTorch: {torch.__version__}")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_accuracy:.4f}"
        )

    metadata = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "final_val_loss": history["val_loss"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
    }
    save_artifacts(history, metadata)
    torch.save(model.state_dict(), OUTPUT_DIR / "iris_mlp.pt")
    print("Artefactos guardados en results/.")


if __name__ == "__main__":
    main()

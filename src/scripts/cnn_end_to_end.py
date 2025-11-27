"""
cnn_end_to_end.py

End-to-end training script for a CNN on FER2013, using the existing
dictionary-based splits from `preprocessing.create_train_test_dicts`.

- Loads train/val/test numeric dicts
- Wraps them in PyTorch Datasets / DataLoaders
- Defines a simple CNN
- Trains with early stopping on validation accuracy
- Evaluates on the test set with accuracy, F1, confusion matrix, ROC and PR curves

This file is designed so that:
- You can run it directly as a script (python cnn_end_to_end.py)
- Or import its classes / functions from a Jupyter notebook for param search, etc.
"""

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# Config & reproducibility
# -------------------------------------------------------

DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_EPOCHS = 25
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For SPEED (non-deterministic but faster):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True



# -------------------------------------------------------
# Dataset & DataLoaders
# -------------------------------------------------------

class EmotionDataset(Dataset):
    """
    Wraps a numeric FER2013 dict:
        dict[label_str] -> list[np.ndarray of shape (48, 48)]

    and exposes (image_tensor, label_index) pairs for PyTorch.
    """

    def __init__(
        self,
        data_dict: Dict[str, List[np.ndarray]],
        label_to_idx: Dict[str, int],
        transform=None,
    ):
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.transform = transform

        for label_str, img_list in data_dict.items():
            idx = label_to_idx[label_str]
            for arr in img_list:
                # ensure float32 numpy array in [0, 1]
                arr = np.asarray(arr, dtype=np.float32)
                if arr.max() > 1.5:  # safety: if still 0-255, normalize
                    arr = arr / 255.0
                self.samples.append((arr, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_np, label_idx = self.samples[index]
        # convert to tensor with shape (1, 48, 48)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, 48, 48]

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label_idx


def build_transforms():
    """
    Returns:
        train_transform, eval_transform
    Both operate on torch.Tensors of shape (1, H, W).
    """
    # Normalization around 0.5 is sufficient here; you can re-compute dataset stats later
    norm = transforms.Normalize(mean=[0.5], std=[0.5])

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            norm,
        ]
    )

    eval_transform = transforms.Compose(
        [
            norm,
        ]
    )

    return train_transform, eval_transform


def build_dataloaders(
    train_dict: Dict[str, List[np.ndarray]],
    val_dict: Dict[str, List[np.ndarray]],
    test_dict: Dict[str, List[np.ndarray]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 4,
):
    """
    Given numeric dicts for train/val/test, create PyTorch DataLoaders.
    """
    class_names = sorted(train_dict.keys())
    label_to_idx = {label: i for i, label in enumerate(class_names)}

    train_tf, eval_tf = build_transforms()

    train_ds = EmotionDataset(train_dict, label_to_idx, transform=train_tf)
    val_ds = EmotionDataset(val_dict, label_to_idx, transform=eval_tf)
    test_ds = EmotionDataset(test_dict, label_to_idx, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


# -------------------------------------------------------
# Model Definition
# -------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    A small CNN for 48x48 grayscale FER2013 images.
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1x48x48 -> 32x24x24
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 32x24x24 -> 64x12x12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 64x12x12 -> 128x6x6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------------------------------------
# Training & Evaluation
# -------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = outputs.max(1)
        running_correct += (preds == targets).sum().item()
        running_total += targets.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            _, preds = outputs.max(1)
            running_correct += (preds == targets).sum().item()
            running_total += targets.size(0)

            all_logits.append(outputs.detach().cpu())
            all_labels.append(targets.detach().cpu())

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    return epoch_loss, epoch_acc, logits, labels


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # maximize validation accuracy
        factor=0.5,
        patience=3,
    )

    best_state = None
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc


# -------------------------------------------------------
# Metrics & Plotting
# -------------------------------------------------------

def compute_classification_metrics(
    y_true,
    y_pred,
    average: str = "weighted",
):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, f1


def plot_confusion(
    y_true,
    y_pred,
    class_names,
    normalize: bool = False,
    title: str = "CNN Confusion Matrix",
):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title + (" (normalized)" if normalize else ""))
    plt.tight_layout()
    plt.show()


def plot_roc_curves(
    y_true,
    y_score,
    class_names,
    title: str = "CNN ROC Curves (One-vs-Rest)",
):
    """
    y_true: shape (N,) with integer class indices
    y_score: shape (N, num_classes) with probabilities or logits
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n_classes = len(class_names)
    classes_idx = list(range(n_classes))

    # binarize labels for one-vs-rest ROC
    y_true_bin = label_binarize(y_true, classes=classes_idx)

    # if y_score are logits, convert to probabilities
    if y_score.ndim == 2:
        # apply softmax row-wise
        exps = np.exp(y_score - y_score.max(axis=1, keepdims=True))
        y_prob = exps / exps.sum(axis=1, keepdims=True)
    else:
        y_prob = y_score

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label=f"ROC {class_names[i]} (AUC={roc_auc[i]:.3f})",
        )

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        linestyle="--",
        lw=3,
        label=f"micro-average (AUC={roc_auc['micro']:.3f})",
    )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pr_curves(
    y_true,
    y_score,
    class_names,
    title: str = "CNN Precision–Recall Curves (One-vs-Rest)",
):
    """
    y_true: shape (N,) with integer class indices
    y_score: shape (N, num_classes) with probabilities or logits
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n_classes = len(class_names)
    classes_idx = list(range(n_classes))

    # binarize labels
    y_true_bin = label_binarize(y_true, classes=classes_idx)

    # if y_score are logits, convert to probabilities
    if y_score.ndim == 2:
        exps = np.exp(y_score - y_score.max(axis=1, keepdims=True))
        y_prob = exps / exps.sum(axis=1, keepdims=True)
    else:
        y_prob = y_score

    precision = {}
    recall = {}
    avg_precision = {}

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_prob[:, i]
        )
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_prob[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_prob.ravel()
    )
    avg_precision["micro"] = average_precision_score(
        y_true_bin, y_prob, average="micro"
    )

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(
            recall[i],
            precision[i],
            lw=2,
            label=f"PR {class_names[i]} (AP={avg_precision[i]:.3f})",
        )

    plt.plot(
        recall["micro"],
        precision["micro"],
        linestyle="--",
        lw=3,
        label=f"micro-average (AP={avg_precision['micro']:.3f})",
    )

    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# End-to-end main
# -------------------------------------------------------

def main():
    """
    End-to-end entry point:
      - Load dict splits
      - Build DataLoaders
      - Train CNN
      - Evaluate on test set and plot metrics
    """
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Lazy import here to avoid circular issues when used from notebooks
    from preprocessing.create_train_test_dicts import build_splits

    # You can adjust val_ratio and seed here to match your KNN experiments
    splits = build_splits(val_ratio=0.25, seed=RANDOM_SEED)

    train_dict = splits["train_numeric"]
    val_dict = splits["val_numeric"]
    test_dict = splits["test_numeric"]

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        train_dict, val_dict, test_dict, batch_size=DEFAULT_BATCH_SIZE
    )

    num_classes = len(class_names)
    model = SimpleCNN(num_classes=num_classes).to(device)

    print(model)

    model, best_val_acc = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        num_epochs=DEFAULT_NUM_EPOCHS,
        lr=DEFAULT_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )

    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Final evaluation on test set
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_logits, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Predictions & metrics
    y_true = test_labels
    y_logits = test_logits
    y_pred = y_logits.argmax(axis=1)

    acc, f1 = compute_classification_metrics(y_true, y_pred, average="weighted")
    print(f"Test Accuracy (sklearn): {acc:.4f}")
    print(f"Test F1-weighted: {f1:.4f}")

    # Confusion matrix
    plot_confusion(y_true, y_pred, class_names, normalize=False)
    plot_confusion(y_true, y_pred, class_names, normalize=True)

    # ROC & PR curves (one-vs-rest)
    plot_roc_curves(y_true, y_logits, class_names)
    plot_pr_curves(y_true, y_logits, class_names)

def run_cnn_param_search(
    lrs,
    weight_decays,
    num_epochs=5,
    batch_size=128,
    val_ratio=0.25,
    seed=42,
):
    """
    Grid search over (learning rate, weight_decay) for SimpleCNN.
    Returns a pandas DataFrame with one row per combination.
    """

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- load splits once ---
    splits = build_splits(val_ratio=val_ratio, seed=seed)
    train_dict = splits["train_numeric"]
    val_dict   = splits["val_numeric"]
    test_dict  = splits["test_numeric"]  # not used in search, only train+val

    # --- build dataloaders once (batch_size fixed for this search) ---
    train_loader, val_loader, _, class_names = build_dataloaders(
        train_dict, val_dict, test_dict,
        batch_size=batch_size,
        num_workers=4,
    )

    results = []

    for lr, wd in itertools.product(lrs, weight_decays):
        print(f"\n=== lr={lr}, weight_decay={wd} ===")

        # fresh model each time
        model = SimpleCNN(num_classes=len(class_names)).to(device)

        # train for a small number of epochs (fast-ish search)
        model, best_val_acc = train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=wd,
        )

        # optional: evaluate once more on val to be sure
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"[FINAL] lr={lr}, wd={wd} | best_val_acc={best_val_acc:.4f} | val_acc={val_acc:.4f}")

        results.append(
            {
                "lr": lr,
                "weight_decay": wd,
                "best_val_acc": float(best_val_acc),
                "val_acc_end": float(val_acc),
            }
        )

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    main()

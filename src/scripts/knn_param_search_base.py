import numpy as np
from typing import Iterable, Dict, Tuple, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def evaluate_k_values(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k_values: Optional[Iterable[int]] = None,
    weights: str = "distance",
):
    if k_values is None:
        k_values = list(range(1, 20, 2))

    k_values = list(k_values)
    scores: Dict[int, float] = {}

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, weights=weights)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        scores[k] = float(acc)
        print(f"k={k:2d} | Val accuracy = {acc:.4f}")

    best_k = max(scores, key=scores.get)
    best_acc = scores[best_k]

    print(f"\nBest k: {best_k} with accuracy {best_acc:.4f}\n")

    return best_k, best_acc, scores


def plot_k_scores(scores: Dict[int, float]):
    import matplotlib.pyplot as plt

    ks = list(scores.keys())
    accs = list(scores.values())

    plt.figure(figsize=(8,4))
    plt.plot(ks, accs, marker='o')
    plt.xlabel("k (n_neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.title("KNN Hyperparameter Search")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

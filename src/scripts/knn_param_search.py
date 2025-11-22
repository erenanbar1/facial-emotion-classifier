import numpy as np
from typing import Dict, Iterable, List, Optional, Tuple

from sklearn.decomposition import PCA

# Reuse your existing evaluation logic
from scripts.knn_param_search_base import evaluate_k_values


# --------------------------------------------------
# Helper: Run PCA
# --------------------------------------------------
def apply_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    n_components: int,
    whiten: bool = True,
    random_state: int = 42,
):
    """
    Fit PCA on X_train and transform both X_train and X_val.
    Returns (X_train_pca, X_val_pca, pca_object).
    """
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    pca.fit(X_train)
    return pca.transform(X_train), pca.transform(X_val), pca


# --------------------------------------------------
# MAIN GRID SEARCH: PCA dims × K values
# --------------------------------------------------
def knn_pca_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pca_dims: Iterable[int] = (50, 100, 150),
    k_values: Iterable[int] = range(1, 31),
    weights: str = "distance",
) -> Dict[str, object]:
    """
    Explore KNN performance over a grid of:
      - PCA dimensions
      - K (n_neighbors) values

    Args:
        X_train, y_train: training data (before PCA)
        X_val, y_val: validation data (before PCA)
        pca_dims: iterable of PCA dimensions to try
        k_values: iterable of K values to try
        weights: 'uniform' or 'distance' (passed to KNeighborsClassifier)

    Returns:
        results: dict with fields:
            - 'grid': list of dicts, each:
                  {'pca_dim': int, 'k': int, 'acc': float}
            - 'per_pca_best': list of dicts, each:
                  {'pca_dim': int, 'best_k': int, 'best_acc': float}
            - 'global_best': dict:
                  {'pca_dim': int, 'k': int, 'acc': float}
    """
    pca_dims = list(pca_dims)
    k_values = list(k_values)

    grid: List[Dict[str, object]] = []
    per_pca_best: List[Dict[str, object]] = []

    global_best_acc = -1.0
    global_best_combo: Dict[str, object] = {}

    for pca_dim in pca_dims:
        print(f"\n=== PCA dim = {pca_dim} ===")

        # 1) Apply PCA for this dimension
        X_train_pca, X_val_pca, _ = apply_pca(X_train, X_val, n_components=pca_dim)

        # 2) Use existing helper to evaluate all K for this PCA dim
        best_k, best_acc, scores = evaluate_k_values(
            X_train_pca,
            y_train,
            X_val_pca,
            y_val,
            k_values=k_values,
            weights=weights,
        )

        # 3) Store per-K results into the grid
        for k, acc in scores.items():
            grid.append({"pca_dim": pca_dim, "k": k, "acc": acc})

        # 4) Keep track of best per PCA dim
        per_pca_best.append(
            {"pca_dim": pca_dim, "best_k": best_k, "best_acc": best_acc}
        )

        # 5) Track global best over all PCA dims and Ks
        if best_acc > global_best_acc:
            global_best_acc = best_acc
            global_best_combo = {"pca_dim": pca_dim, "k": best_k, "acc": best_acc}

    results = {
        "grid": grid,
        "per_pca_best": per_pca_best,
        "global_best": global_best_combo,
    }
    return results


# --------------------------------------------------
# Compact plot: multiple PCA curves on same axes
# --------------------------------------------------
def plot_knn_pca_grid(results: Dict[str, object], title: str = "KNN Hyperparameter Search"):
    """
    Plot K (x-axis) vs validation accuracy (y-axis) for each PCA dimension
    in a single compact figure, similar to your example image.

    Expects the result dict returned by knn_pca_grid_search.
    """
    import matplotlib.pyplot as plt

    grid: List[Dict[str, object]] = results["grid"]
    # organize scores by pca_dim -> {k: acc}
    by_dim: Dict[int, Dict[int, float]] = {}

    for row in grid:
        dim = int(row["pca_dim"])
        k   = int(row["k"])
        acc = float(row["acc"])
        by_dim.setdefault(dim, {})[k] = acc

    plt.figure(figsize=(10, 4))

    for dim in sorted(by_dim.keys()):
        k_list = sorted(by_dim[dim].keys())
        acc_list = [by_dim[dim][k] for k in k_list]
        plt.plot(k_list, acc_list, marker="o", label=f"PCA={dim}")

    plt.xlabel("k (n_neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.legend(title="PCA dim", loc="best")
    plt.tight_layout()
    plt.show()

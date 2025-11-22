import numpy as np
from typing import Iterable, Dict, Tuple, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_k_values(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k_values: Optional[Iterable[int]] = None,
    weights: str = "distance",
    metric: str = "accuracy",
) -> Tuple[int, float, Dict[int, float]]:
    """
    Evaluate KNN for multiple K values using a selected metric.
    
    metric options:
      - "accuracy"
      - "f1_macro"
      - "f1_weighted"
      - "precision_macro"
      - "precision_weighted"
      - "recall_macro"
      - "recall_weighted"
    """

    if k_values is None:
        k_values = list(range(1, 20, 2))

    # choose metric function
    if metric == "accuracy":
        metric_fn = accuracy_score
    elif metric == "f1_macro":
        metric_fn = lambda yt, yp: f1_score(yt, yp, average="macro")
    elif metric == "f1_weighted":
        metric_fn = lambda yt, yp: f1_score(yt, yp, average="weighted")
    elif metric == "precision_macro":
        metric_fn = lambda yt, yp: precision_score(yt, yp, average="macro")
    elif metric == "precision_weighted":
        metric_fn = lambda yt, yp: precision_score(yt, yp, average="weighted")
    elif metric == "recall_macro":
        metric_fn = lambda yt, yp: recall_score(yt, yp, average="macro")
    elif metric == "recall_weighted":
        metric_fn = lambda yt, yp: recall_score(yt, yp, average="weighted")
    else:
        raise ValueError(f"Unknown metric '{metric}'")

    scores: Dict[int, float] = {}

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, weights=weights)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        score = float(metric_fn(y_val, y_pred))
        scores[k] = score

        #print(f"k={k:2d} | {metric} = {score:.4f}")

    # pick best k
    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]

    print(f"\nBest k based on {metric}: {best_k} (score={best_score:.4f})")

    return best_k, best_score, scores



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

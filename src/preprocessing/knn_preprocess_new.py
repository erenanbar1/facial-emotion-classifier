import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def dict_to_xy_flat(train_dict):
    """
    Convert a dict[label] -> list of 48x48 images
    into:
        X: (N, 2304) normalized float32
        y: (N,) labels
    """
    X_list = []
    y_list = []

    for label, img_list in train_dict.items():
        for img in img_list:
            arr = img.astype("float32")
            if arr.max() > 1.5:   # assume 0-255 → normalize
                arr = arr / 255.0
            flat = arr.flatten()  # (2304,)
            X_list.append(flat)
            y_list.append(label)

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list)
    return X, y

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def fit_knn_from_dict(train_dict, n_components=50, n_neighbors=8):
    """
    Full pipeline:
      dict -> X,y -> scale -> PCA(whiten) -> KNN fit

    Returns:
      knn     : trained KNeighborsClassifier
      scaler  : fitted StandardScaler
      pca     : fitted PCA
      X_train_pca : (N_train, n_components) features
      y_train     : labels
    """
    # 1) dict -> X,y (normalized & flattened)
    X_train, y_train = dict_to_xy_flat(train_dict)

    # 2) Standardize
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)

    # 3) PCA + whitening
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train_std)

    # 4) KNN
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",     # what you’re using already
        metric="euclidean"
    )
    knn.fit(X_train_pca, y_train)

    return knn, scaler, pca, X_train_pca, y_train

def dict_to_features_with_fitted(train_like_dict, scaler, pca):
    """
    Use already-fitted scaler + pca to transform
    another split dict (val/test) into PCA-whitened features.

    Returns:
      X_pca: (N, n_components)
      y    : labels
    """
    X, y = dict_to_xy_flat(train_like_dict)
    X_std = scaler.transform(X)
    X_pca = pca.transform(X_std)
    return X_pca, y

from sklearn.metrics import accuracy_score, f1_score  # and others if you want


class KNNParamSearch:
    def __init__(self, train_dict, val_dict, pca_dims, k_values, metrics):
        """
        train_dict, val_dict: split dicts in numeric 48x48 form
                              like your train_split / val_split

        pca_dims:  iterable of PCA dimensions to try, e.g. [25, 50, 75]
        k_values:  iterable of k values to try, e.g. [3, 5, 8, 11]

        metrics:   dict[str, callable],
                   e.g. {
                     "accuracy": accuracy_score,
                     "f1_weighted": lambda y_true, y_pred:
                         f1_score(y_true, y_pred, average="weighted")
                   }
        """
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.pca_dims = list(pca_dims)
        self.k_values = list(k_values)
        self.metrics = metrics  # {"name": func}
        # ---------- NEW: train final model on train+val using best params ----------
    def train_best_model(self, val_results, test_dict, metric_name="f1_weighted"):
        """
        1) Select best (pca_dim, k) from validation results.
        2) Merge train + val.
        3) Train final KNN on train+val with best hyperparams.
        4) Evaluate on test_dict and return everything needed for plots.

        Returns:
            knn_final, scaler_final, pca_final,
            X_test_pca, y_test, y_score, class_names
        """
        # use your existing helper
        best_pca_dim, best_k, best_row = select_best_params(val_results, metric_name)

        # merge train + val
        train_val_split = merge_split_dicts(self.train_dict, self.val_dict)

        # fit final model on train+val
        knn_final, scaler_final, pca_final, X_trainval_pca, y_trainval = fit_knn_from_dict(
            train_val_split,
            n_components=best_pca_dim,
            n_neighbors=best_k,
        )

        # transform test set
        X_test_pca, y_test = dict_to_features_with_fitted(
            test_dict,
            scaler_final,
            pca_final,
        )

        # predict probabilities for ROC/PR
        y_score = knn_final.predict_proba(X_test_pca)

        # class names in consistent order
        class_names = np.unique(y_trainval)

        print(f"[Final model] PCA={best_pca_dim}, k={best_k}")
        print(f"Train+Val size: {len(y_trainval)}, Test size: {len(y_test)}")

        return knn_final, scaler_final, pca_final, X_test_pca, y_test, y_score, class_names

    # ---------- NEW: ROC curve plotting ----------
    @staticmethod
    def plot_roc_curves(y_test, y_score, class_names, title="KNN ROC Curves (One-vs-Rest)"):
        """
        Plot one-vs-rest ROC curves + micro-average ROC.
        """
        n_classes = len(class_names)

        # binarize labels for OVR ROC
        y_test_bin = label_binarize(y_test, classes=class_names)

        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # micro-average
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # plot
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.plot(
                fpr[i],
                tpr[i],
                lw=2,
                label=f"ROC {class_names[i]} (AUC={roc_auc[i]:.3f})"
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

    # ---------- NEW: Precision–Recall curve plotting ----------
    @staticmethod
    def plot_pr_curves(y_test, y_score, class_names, title="KNN Precision–Recall Curves (One-vs-Rest)"):
        """
        Plot one-vs-rest Precision–Recall curves + micro-average PR.
        """
        n_classes = len(class_names)

        # binarize labels
        y_test_bin = label_binarize(y_test, classes=class_names)

        precision = {}
        recall = {}
        avg_precision = {}

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_test_bin[:, i],
                y_score[:, i],
            )
            avg_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

        # micro-average
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test_bin.ravel(),
            y_score.ravel(),
        )
        avg_precision["micro"] = average_precision_score(
            y_test_bin,
            y_score,
            average="micro",
        )

        # plot
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.plot(
                recall[i],
                precision[i],
                lw=2,
                label=f"PR {class_names[i]} (AP={avg_precision[i]:.3f})"
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

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, title="Confusion Matrix"):
        """
        Plot confusion matrix using seaborn heatmap.

        normalize:
            False → raw counts
            True  → normalize rows to percentages
        """
        if normalize:
            cm = confusion_matrix(y_true, y_pred, labels=class_names, normalize="true")
        else:
            cm = confusion_matrix(y_true, y_pred, labels=class_names)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        plt.title(title)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.show()

    def run(self):
        """
        Run grid search over all (pca_dim, k) pairs.

        Returns:
            results: list of dicts, one per (pca_dim, k), e.g.:

            [
              {"pca_dim": 25, "k": 3, "accuracy": 0.61, "f1_weighted": 0.59},
              {"pca_dim": 25, "k": 5, "accuracy": 0.62, "f1_weighted": 0.60},
              ...
            ]
        """
        results = []

        for pca_dim in self.pca_dims:
            for k in self.k_values:
                print(f"Evaluating: PCA={pca_dim}, k={k}")

                # 1) fit on train
                knn, scaler, pca, X_train_pca, y_train = fit_knn_from_dict(
                    self.train_dict,
                    n_components=pca_dim,
                    n_neighbors=k,
                )

                # 2) transform val
                X_val_pca, y_val = dict_to_features_with_fitted(
                    self.val_dict,
                    scaler,
                    pca,
                )

                # 3) predict
                y_pred = knn.predict(X_val_pca)

                # 4) compute metrics
                row = {
                    "pca_dim": pca_dim,
                    "k": k,
                }
                for name, metric_fn in self.metrics.items():
                    score = metric_fn(y_val, y_pred)
                    row[name] = score

                results.append(row)

        return results
    
def merge_split_dicts(dict_a, dict_b):
    """
    Merge two label -> list-of-images dictionaries.

    For each label, concatenates the lists from both dicts.
    Assumes the same label set (but is robust if one dict is missing a label).
    """
    merged = {}

    # all labels that appear in either dict
    all_labels = set(dict_a.keys()) | set(dict_b.keys())

    for label in all_labels:
        imgs_a = dict_a.get(label, [])
        imgs_b = dict_b.get(label, [])
        merged[label] = list(imgs_a) + list(imgs_b)

    return merged

import pandas as pd

def select_best_params(results, metric_name="f1_weighted"):
    """
    results: list of dicts returned from KNNParamSearch.run()
             each dict has keys like: pca_dim, k, accuracy, f1_weighted...

    metric_name: which metric to optimize ("f1_weighted", "accuracy", etc.)

    Returns:
        best_pca_dim, best_k, best_row (as a pandas Series)
    """
    df = pd.DataFrame(results)

    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in results columns: {df.columns}")

    best_idx = df[metric_name].idxmax()
    best_row = df.loc[best_idx]

    best_pca_dim = int(best_row["pca_dim"])
    best_k = int(best_row["k"])

    print(f"Best by {metric_name}: PCA={best_pca_dim}, k={best_k}, score={best_row[metric_name]:.4f}")
    return best_pca_dim, best_k, best_row





import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def dict_to_xy_flat(data_dict):
    """
    Convert a dict[label] -> list of 48x48 images into flat feature vectors for SVM.

    Returns:
        X: (N, 2304) float32, normalized to [0,1] if inputs look like 0–255
        y: (N,) labels
    """
    X_list = []
    y_list = []
    for label, images in data_dict.items():
        for img in images:
            arr = img.astype("float32")
            if arr.max() > 1.5:  # assume uint8 range, normalize
                arr = arr / 255.0
            X_list.append(arr.flatten())
            y_list.append(label)
    X = np.array(X_list, dtype="float32")
    y = np.array(y_list)
    return X, y


def fit_svc_pipeline(
    train_dict,
    n_components=None,
    C=1.0,
    gamma="scale",
    probability=True,
    shrinking=True,
    tol=1e-3,
    cache_size=200,
    verbose=False,
):
    """
    Flatten -> Standardize -> optional PCA -> SVC fit (libsvm-based).

    Returns:
        svc, scaler, pca (or None), X_train_transformed, y_train
    """
    print(f"Fitting SVC pipeline:  C={C}, gamma={gamma}, n_components={n_components}")
    X_train, y_train = dict_to_xy_flat(train_dict)
    print(f"  Train size: {len(y_train)}")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_train)

    pca = None
    X_feats = X_std
    if n_components is not None:
        pca = PCA(n_components=n_components, whiten=False, random_state=42)
        X_feats = pca.fit_transform(X_std)
    print(f"  Feature dim after PCA: {X_feats.shape[1]}")
    svc = SVC(
        C=C,
        kernel="rbf",
        gamma=gamma if gamma is not None else "scale",
        shrinking=shrinking,
        probability=probability,
        cache_size=cache_size,
        random_state=42,
        verbose=verbose,
    )
    print("  Training SVC...")
    svc.fit(X_feats, y_train)
    print("  SVC training complete.")
    return svc, scaler, pca, X_feats, y_train


def transform_with_fitted(data_dict, scaler, pca=None):
    """
    Apply fitted scaler/PCA to another split and return (X_transformed, y).
    """
    X, y = dict_to_xy_flat(data_dict)
    X_std = scaler.transform(X)
    if pca is not None:
        X_std = pca.transform(X_std)
    return X_std, y


class SVCParamSearch:
    """
    Lightweight grid search wrapper focusing only on C and gamma (RBF kernel).
    """

    def __init__(
        self,
        train_dict,
        val_dict,
        pca_dims,
        C_values,
        gamma_values,
        metrics=None,
        probability=False,
        svc_verbose=False,
    ):
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.pca_dims = list(pca_dims)
        self.C_values = list(C_values)
        self.gamma_values = list(gamma_values)
        self.metrics = metrics or {
            "accuracy": accuracy_score,
            "f1_weighted": lambda y_true, y_pred: f1_score(
                y_true, y_pred, average="weighted"
            ),
        }
        self.probability = probability
        self.svc_verbose = svc_verbose

    def run(self):
        results = []
        for pca_dim in self.pca_dims:
            for C in self.C_values:
                for gamma in self.gamma_values:
                    desc = f"PCA={pca_dim}, kernel=rbf, C={C}, gamma={gamma}"
                    print(f"Evaluating: {desc}")

                    svc, scaler, pca, _, _ = fit_svc_pipeline(
                        self.train_dict,
                        n_components=pca_dim,
                        C=C,
                        gamma=gamma,
                        probability=self.probability,
                        verbose=self.svc_verbose,
                    )
                    print(f"  [Going SVC] {desc}")
                    X_val, y_val = transform_with_fitted(
                        self.val_dict,
                        scaler,
                        pca,
                    )
                    print(f"  Val size: {len(y_val)}")
                    y_pred = svc.predict(X_val)
                    print(f"  Predictions done.")
                    row = {
                        "pca_dim": pca_dim,
                        "kernel": "rbf",
                        "C": C,
                        "gamma": gamma,
                    }
                    for name, metric_fn in self.metrics.items():
                        row[name] = metric_fn(y_val, y_pred)
                    print(
                        "  Results: "
                        + ", ".join(
                            [
                                f"{k}={v}"
                                for k, v in row.items()
                                if k
                                not in ("pca_dim", "kernel", "C", "gamma", "degree", "coef0", "class_weight")
                            ]
                        )
                    )
                    results.append(row)
                    print("")
        return results

    def train_best_model(self, val_results, test_dict, metric_name="f1_weighted"):
        """
        Pick best params by metric, retrain on train+val, evaluate on test.
        """
        best_params, best_row = select_best(val_results, metric_name)

        train_val_dict = merge_dicts(self.train_dict, self.val_dict)
        svc_final, scaler_final, pca_final, _, y_trainval = fit_svc_pipeline(
            train_val_dict,
            n_components=best_params["pca_dim"],
            C=best_params["C"],
            gamma=best_params["gamma"],
            probability=self.probability,
            verbose=self.svc_verbose,
        )

        X_test, y_test = transform_with_fitted(
            test_dict,
            scaler_final,
            pca_final,
        )

        if self.probability:
            y_score = svc_final.predict_proba(X_test)
        else:
            y_score = svc_final.decision_function(X_test)

        class_names = np.unique(y_trainval)
        print(
            f"[Final SVC] PCA={best_params['pca_dim']}, kernel=rbf, "
            f"C={best_params['C']}, gamma={best_params['gamma']}"
        )
        print(f"Train+Val size: {len(y_trainval)}, Test size: {len(y_test)}")

        return svc_final, scaler_final, pca_final, X_test, y_test, y_score, class_names

    # ---------- Plotting ----------
    @staticmethod
    def plot_roc(y_test, y_score, class_names, title="SVC ROC (OvR)"):
        n_classes = len(class_names)
        y_bin = label_binarize(y_test, classes=class_names)
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
        plt.plot(fpr["micro"], tpr["micro"], "--", lw=3, label=f"micro (AUC={roc_auc['micro']:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.title(title)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_pr(y_test, y_score, class_names, title="SVC Precision–Recall (OvR)"):
        n_classes = len(class_names)
        y_bin = label_binarize(y_test, classes=class_names)
        precision, recall, ap = {}, {}, {}
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
            ap[i] = average_precision_score(y_bin[:, i], y_score[:, i])
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_bin.ravel(), y_score.ravel())
        ap["micro"] = average_precision_score(y_bin, y_score, average="micro")

        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.plot(recall[i], precision[i], lw=2, label=f"{class_names[i]} (AP={ap[i]:.3f})")
        plt.plot(recall["micro"], precision["micro"], "--", lw=3, label=f"micro (AP={ap['micro']:.3f})")
        plt.title(title)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cm(y_true, y_pred, class_names, normalize=False, title="Confusion Matrix"):
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=class_names,
            normalize="true" if normalize else None,
        )
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
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()


def merge_dicts(a, b):
    merged = {}
    labels = set(a.keys()) | set(b.keys())
    for label in labels:
        merged[label] = list(a.get(label, [])) + list(b.get(label, []))
    return merged


def select_best(results, metric_name="f1_weighted"):
    df = pd.DataFrame(results)
    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in results columns: {df.columns}")
    best_idx = df[metric_name].idxmax()
    best_row = df.loc[best_idx]
    best_params = {
        "pca_dim": best_row["pca_dim"],
        "C": best_row["C"],
        "gamma": best_row["gamma"],
    }
    print(
        f"Best by {metric_name}: PCA={best_params['pca_dim']}, kernel=rbf, "
        f"C={best_params['C']}, gamma={best_params['gamma']}, "
        f"score={best_row[metric_name]:.4f}"
    )
    return best_params, best_row

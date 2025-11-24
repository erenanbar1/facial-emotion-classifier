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
    Convert a dict[label] -> list of 48x48 images into flat feature vectors.

    Returns:
        X: (N, 2304) float32, normalized to [0,1] if needed
        y: (N,) labels
    """
    X_list = []
    y_list = []

    for label, images in data_dict.items():
        for img in images:
            arr = img.astype("float32")
            if arr.max() > 1.5:  # assume 0-255 → normalize
                arr = arr / 255.0
            X_list.append(arr.flatten())
            y_list.append(label)

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list)
    return X, y


def fit_svm_from_dict(
    train_dict,
    n_components=None,
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight=None,
    probability=True,
):
    """
    Full pipeline: flatten -> standardize -> optional PCA -> SVM fit.

    Returns:
        svm, scaler, pca (or None), X_train_transformed, y_train
    """
    X_train, y_train = dict_to_xy_flat(train_dict)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_train)

    pca = None
    X_features = X_std
    if n_components is not None:
        pca = PCA(n_components=n_components, whiten=False, random_state=42)
        X_features = pca.fit_transform(X_std)

    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        probability=probability,
        random_state=42,
    )
    svm.fit(X_features, y_train)
    return svm, scaler, pca, X_features, y_train


def dict_to_features_with_fitted(data_dict, scaler, pca=None):
    """
    Transform another split (val/test) using fitted scaler/PCA.

    Returns:
        X_transformed, y
    """
    X, y = dict_to_xy_flat(data_dict)
    X_std = scaler.transform(X)
    if pca is not None:
        X_std = pca.transform(X_std)
    return X_std, y


class SVMParamSearch:
    def __init__(
        self,
        train_dict,
        val_dict,
        pca_dims,
        kernels,
        C_values,
        gamma_values,
        class_weights=(None, "balanced"),
        metrics=None,
        probability=True,
    ):
        """
        Hyperparameter search for SVM over PCA dims, kernels, C, gamma, and class_weight.

        metrics: dict[name -> callable], e.g. accuracy_score, f1_weighted, macro_f1, etc.
        """
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.pca_dims = list(pca_dims)
        self.kernels = list(kernels)
        self.C_values = list(C_values)
        self.gamma_values = list(gamma_values)
        self.class_weights = list(class_weights)
        self.metrics = metrics or {
            "accuracy": accuracy_score,
            "f1_weighted": lambda y_true, y_pred: f1_score(
                y_true, y_pred, average="weighted"
            ),
        }
        self.probability = probability

    def run(self):
        results = []
        for pca_dim in self.pca_dims:
            for kernel in self.kernels:
                for C in self.C_values:
                    for cw in self.class_weights:
                        if kernel == "linear":
                            gamma_list = [None]  # gamma ignored for linear; marker
                        else:
                            gamma_list = self.gamma_values

                        for gamma in gamma_list:
                            msg = f"PCA={pca_dim}, kernel={kernel}, C={C}, cw={cw}, gamma={gamma}"
                            print(f"Evaluating: {msg}")

                            svm, scaler, pca, _, _ = fit_svm_from_dict(
                                self.train_dict,
                                n_components=pca_dim,
                                kernel=kernel,
                                C=C,
                                gamma="scale" if gamma is None else gamma,
                                class_weight=cw,
                                probability=self.probability,
                            )

                            X_val, y_val = dict_to_features_with_fitted(
                                self.val_dict,
                                scaler,
                                pca,
                            )
                            y_pred = svm.predict(X_val)

                            row = {
                                "pca_dim": pca_dim,
                                "kernel": kernel,
                                "C": C,
                                "gamma": gamma,
                                "class_weight": cw,
                            }
                            for name, metric_fn in self.metrics.items():
                                row[name] = metric_fn(y_val, y_pred)
                            results.append(row)
        return results

    def train_best_model(
        self,
        val_results,
        test_dict,
        metric_name="f1_weighted",
    ):
        """
        Select best params by metric, retrain on train+val, evaluate on test.

        Returns:
            svm_final, scaler_final, pca_final, X_test_transformed, y_test, y_score, class_names
        """
        best_params, best_row = select_best_params(val_results, metric_name)

        train_val_dict = merge_split_dicts(self.train_dict, self.val_dict)
        svm_final, scaler_final, pca_final, _, y_trainval = fit_svm_from_dict(
            train_val_dict,
            n_components=best_params["pca_dim"],
            kernel=best_params["kernel"],
            C=best_params["C"],
            gamma="scale" if best_params["gamma"] is None else best_params["gamma"],
            class_weight=best_params["class_weight"],
            probability=self.probability,
        )

        X_test, y_test = dict_to_features_with_fitted(
            test_dict,
            scaler_final,
            pca_final,
        )

        # decision_function works for ROC/PR; if probability=True we can use predict_proba
        if self.probability:
            y_score = svm_final.predict_proba(X_test)
        else:
            y_score = svm_final.decision_function(X_test)

        class_names = np.unique(y_trainval)

        print(
            f"[Final SVM] PCA={best_params['pca_dim']}, kernel={best_params['kernel']}, "
            f"C={best_params['C']}, gamma={best_params['gamma']}, cw={best_params['class_weight']}"
        )
        print(f"Train+Val size: {len(y_trainval)}, Test size: {len(y_test)}")

        return svm_final, scaler_final, pca_final, X_test, y_test, y_score, class_names

    # ---------- Plotting helpers ----------
    @staticmethod
    def plot_roc_curves(y_test, y_score, class_names, title="SVM ROC Curves (One-vs-Rest)"):
        n_classes = len(class_names)
        y_test_bin = label_binarize(y_test, classes=class_names)

        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
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

    @staticmethod
    def plot_pr_curves(y_test, y_score, class_names, title="SVM Precision–Recall Curves (One-vs-Rest)"):
        n_classes = len(class_names)
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

        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test_bin.ravel(),
            y_score.ravel(),
        )
        avg_precision["micro"] = average_precision_score(
            y_test_bin,
            y_score,
            average="micro",
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

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, title="Confusion Matrix"):
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


def merge_split_dicts(dict_a, dict_b):
    merged = {}
    all_labels = set(dict_a.keys()) | set(dict_b.keys())
    for label in all_labels:
        merged[label] = list(dict_a.get(label, [])) + list(dict_b.get(label, []))
    return merged


def select_best_params(results, metric_name="f1_weighted"):
    df = pd.DataFrame(results)
    if metric_name not in df.columns:
        raise ValueError(
            f"Metric '{metric_name}' not found in results columns: {df.columns}"
        )

    best_idx = df[metric_name].idxmax()
    best_row = df.loc[best_idx]

    best_params = {
        "pca_dim": best_row["pca_dim"],
        "kernel": best_row["kernel"],
        "C": best_row["C"],
        "gamma": best_row["gamma"],
        "class_weight": best_row["class_weight"],
    }

    print(
        f"Best by {metric_name}: "
        f"PCA={best_params['pca_dim']}, kernel={best_params['kernel']}, "
        f"C={best_params['C']}, gamma={best_params['gamma']}, cw={best_params['class_weight']}, "
        f"score={best_row[metric_name]:.4f}"
    )
    return best_params, best_row

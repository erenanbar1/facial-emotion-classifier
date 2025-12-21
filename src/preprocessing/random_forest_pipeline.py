import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
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


def fit_rf_from_dict(
    train_dict,
    n_components=None,
    n_estimators=300,
    max_depth=None,
    max_features="sqrt",
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight=None,
    n_jobs=-1,
    random_state=42,
):
    """
    Full pipeline: flatten -> optional PCA -> RandomForest fit.

    Returns:
        rf, pca (or None), X_train_transformed, y_train
    """
    X_train, y_train = dict_to_xy_flat(train_dict)

    pca = None
    X_features = X_train
    if n_components is not None:
        pca = PCA(n_components=n_components, whiten=False, random_state=random_state)
        X_features = pca.fit_transform(X_train)

    if isinstance(max_depth, float) and np.isnan(max_depth):
        max_depth = None

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    rf.fit(X_features, y_train)
    return rf, pca, X_features, y_train


def dict_to_features_with_fitted(data_dict, pca=None):
    """
    Transform another split (val/test) using fitted PCA (if provided).

    Returns:
        X_transformed, y
    """
    X, y = dict_to_xy_flat(data_dict)
    if pca is not None:
        X = pca.transform(X)
    return X, y


class RFParamSearch:
    def __init__(
        self,
        train_dict,
        val_dict,
        pca_dims,
        n_estimators_list,
        max_depth_list,
        max_features_list,
        min_samples_split_list=(2,),
        min_samples_leaf_list=(1,),
        class_weights=(None,),
        metrics=None,
        random_state=42,
        n_jobs=-1,
    ):
        """
        Hyperparameter search for RandomForest over PCA dims, estimators, depth, etc.

        metrics: dict[name -> callable], e.g. accuracy_score, f1_weighted, macro_f1
        """
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.pca_dims = list(pca_dims)
        self.n_estimators_list = list(n_estimators_list)
        self.max_depth_list = list(max_depth_list)
        self.max_features_list = list(max_features_list)
        self.min_samples_split_list = list(min_samples_split_list)
        self.min_samples_leaf_list = list(min_samples_leaf_list)
        self.class_weights = list(class_weights)
        self.metrics = metrics or {
            "accuracy": accuracy_score,
            "f1_weighted": lambda y_true, y_pred: f1_score(
                y_true, y_pred, average="weighted"
            ),
        }
        self.random_state = random_state
        self.n_jobs = n_jobs

    def run(self):
        results = []
        min_split = self.min_samples_split_list[0]
        min_leaf = self.min_samples_leaf_list[0]
        cw = self.class_weights[0]
        for pca_dim in self.pca_dims:
            for n_estimators in self.n_estimators_list:
                for max_depth in self.max_depth_list:
                    for max_features in self.max_features_list:
                        msg = (
                            f"PCA={pca_dim}, n_estimators={n_estimators}, "
                            f"max_depth={max_depth}, max_features={max_features}, "
                            f"min_split={min_split}, min_leaf={min_leaf}, cw={cw}"
                        )
                        print(f"Evaluating: {msg}")

                        rf, pca, _, _ = fit_rf_from_dict(
                            self.train_dict,
                            n_components=pca_dim,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            max_features=max_features,
                            min_samples_split=min_split,
                            min_samples_leaf=min_leaf,
                            class_weight=cw,
                            random_state=self.random_state,
                            n_jobs=self.n_jobs,
                        )

                        X_val, y_val = dict_to_features_with_fitted(
                            self.val_dict,
                            pca,
                        )
                        y_pred = rf.predict(X_val)

                        row = {
                            "pca_dim": pca_dim,
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "max_features": max_features,
                            "min_samples_split": min_split,
                            "min_samples_leaf": min_leaf,
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
            rf_final, pca_final, X_test, y_test, y_score, class_names
        """
        best_params, best_row = select_best_params(val_results, metric_name)

        train_val_dict = merge_split_dicts(self.train_dict, self.val_dict)
        rf_final, pca_final, _, y_trainval = fit_rf_from_dict(
            train_val_dict,
            n_components=best_params["pca_dim"],
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            max_features=best_params["max_features"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            class_weight=best_params["class_weight"],
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        X_test, y_test = dict_to_features_with_fitted(
            test_dict,
            pca_final,
        )

        y_score = rf_final.predict_proba(X_test)
        class_names = np.unique(y_trainval)

        print(
            "[Final RF] "
            f"PCA={best_params['pca_dim']}, n_estimators={best_params['n_estimators']}, "
            f"max_depth={best_params['max_depth']}, max_features={best_params['max_features']}, "
            f"min_split={best_params['min_samples_split']}, min_leaf={best_params['min_samples_leaf']}, "
            f"cw={best_params['class_weight']}"
        )
        print(f"Train+Val size: {len(y_trainval)}, Test size: {len(y_test)}")

        return rf_final, pca_final, X_test, y_test, y_score, class_names

    # ---------- Plotting helpers ----------
    @staticmethod
    def plot_roc_curves(y_test, y_score, class_names, title="RF ROC Curves (One-vs-Rest)"):
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
    def plot_pr_curves(y_test, y_score, class_names, title="RF Precision–Recall (One-vs-Rest)"):
        n_classes = len(class_names)
        y_test_bin = label_binarize(y_test, classes=class_names)

        precision = {}
        recall = {}
        avg_precision = {}

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_test_bin[:, i], y_score[:, i]
            )
            avg_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test_bin.ravel(), y_score.ravel()
        )
        avg_precision["micro"] = average_precision_score(
            y_test_bin, y_score, average="micro"
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

    max_depth = best_row["max_depth"]
    if pd.isna(max_depth):
        max_depth = None

    best_params = {
        "pca_dim": best_row["pca_dim"],
        "n_estimators": best_row["n_estimators"],
        "max_depth": max_depth,
        "max_features": best_row["max_features"],
        "min_samples_split": best_row["min_samples_split"],
        "min_samples_leaf": best_row["min_samples_leaf"],
        "class_weight": best_row["class_weight"],
    }

    print(
        f"Best by {metric_name}: "
        f"PCA={best_params['pca_dim']}, n_estimators={best_params['n_estimators']}, "
        f"max_depth={best_params['max_depth']}, max_features={best_params['max_features']}, "
        f"min_split={best_params['min_samples_split']}, min_leaf={best_params['min_samples_leaf']}, "
        f"cw={best_params['class_weight']}, score={best_row[metric_name]:.4f}"
    )
    return best_params, best_row

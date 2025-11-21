import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# --------------------------------------------
# Helper: Convert dictionary → X, y
# --------------------------------------------
def dict_to_xy(data_dict, label_encoder=None):
    """
    Convert a dataset dictionary of the form:
        {"angry": [48x48 arrays...], "happy": [...], ...}
    into:
        X: (N_samples, 2304)
        y: integer encoded labels
        label_encoder: fitted or reused encoder
    """
    X_list = []
    y_str_list = []

    for label, images in data_dict.items():
        for img in images:
            X_list.append(np.asarray(img).reshape(-1))  # flatten 48x48 → 2304
            y_str_list.append(label)

    X = np.stack(X_list)
    y_str = np.array(y_str_list)

    # Fit encoder if first time
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_str)
    else:
        y = label_encoder.transform(y_str)

    return X, y, label_encoder


# --------------------------------------------
# Train KNN
# --------------------------------------------
def train_knn(train_dict, n_neighbors=5):
    """
    Train k-NN from a dictionary dataset.

    Returns:
        knn_model
        label_encoder
    """
    X_train, y_train, le = dict_to_xy(train_dict, label_encoder=None)

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_jobs=-1
    )
    knn.fit(X_train, y_train)

    return knn, le


# --------------------------------------------
# Test KNN & return confusion matrix
# --------------------------------------------
def test_knn(knn_model, label_encoder, test_dict):
    """
    Evaluate model and return confusion matrix + class names.
    """
    X_test, y_true, _ = dict_to_xy(test_dict, label_encoder=label_encoder)
    y_pred = knn_model.predict(X_test)

    cm = confusion_matrix(y_true, y_pred)
    class_names = label_encoder.classes_

    return cm, class_names


# --------------------------------------------
# Find best k with cross-validation
# --------------------------------------------
def find_best_k(train_dict,
                k_values=None,
                cv=5,
                metric="euclidean"):
    """
    Find the best k for KNN using cross-validation on the training set.

    Parameters
    ----------
    train_dict : dict
        Training data dict: {label: [48x48 np.arrays, ...], ...}
    k_values : iterable of int, optional
        Candidate k values. Default = odd numbers from 1 to 21.
    cv : int, optional
        Number of cross-validation folds. Default = 5.
    metric : str, optional
        Distance metric for KNN. Default = "euclidean".

    Returns
    -------
    best_k : int
        k with highest mean CV accuracy.
    best_score : float
        Best mean CV accuracy.
    all_scores : dict
        Mapping k -> mean CV accuracy.
    """
    # dict_to_xy is in the same file, just call it
    X, y, _ = dict_to_xy(train_dict, label_encoder=None)

    if k_values is None:
        k_values = range(1, 22, 2)  # 1,3,5,...,21

    scores = {}

    for k in k_values:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric,
            n_jobs=-1
        )
        cv_scores = cross_val_score(
            knn, X, y,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )
        mean_score = np.mean(cv_scores)
        scores[k] = mean_score
        print(f"k = {k:2d} -> CV accuracy = {mean_score:.4f}")

    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]

    print("\nBest k:", best_k)
    print("Best CV accuracy:", best_score)

    return best_k, best_score, scores

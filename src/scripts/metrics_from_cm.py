import numpy as np

def metrics_from_confusion_matrix(cm):
    """
    Compute accuracy, precision, recall, and F1-score
    from a confusion matrix cm.
    """

    # Total correct predictions = diagonal sum
    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = correct / total

    # Precision, recall, F1 per class
    TP = np.diag(cm)

    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    # Avoid division by zero
    precision = np.where(TP + FP == 0, 0, TP / (TP + FP))
    recall    = np.where(TP + FN == 0, 0, TP / (TP + FN))
    f1        = np.where(
        precision + recall == 0, 
        0,
        2 * (precision * recall) / (precision + recall)
    )

    # Macro averages
    macro_precision = precision.mean()
    macro_recall    = recall.mean()
    macro_f1        = f1.mean()

    return {
        "accuracy": accuracy,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }

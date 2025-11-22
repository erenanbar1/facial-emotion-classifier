import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_image(path):
    """Load a grayscale FER2013 image as float32 array normalized to [0,1]."""
    return np.array(Image.open(path), dtype="float32") / 255.0


def build_dict(root_path):
    """
    Given a root directory such as .../train or .../test,
    return {class_name: [list of file paths]}.
    """
    return {
        class_name: [
            os.path.join(root_path, class_name, fname)
            for fname in os.listdir(os.path.join(root_path, class_name))
        ]
        for class_name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, class_name))
    }


def build_numeric_dict(path_dict):
    """
    Convert {class: [file paths]} → {class: [numpy arrays]}.
    """
    return {
        cls: [load_image(p) for p in paths]
        for cls, paths in path_dict.items()
    }


def split_train_val_sklearn(path_dict, val_ratio=0.25, seed=42):
    """
    Use sklearn.train_test_split for splitting each class list:
      full_train_dict → train_dict + val_dict
    """
    train_split = {}
    val_split = {}

    for cls, paths in path_dict.items():
        train_paths, val_paths = train_test_split(
            paths,
            test_size=val_ratio,
            shuffle=True,
            random_state=seed
        )

        train_split[cls] = train_paths
        val_split[cls] = val_paths

    return train_split, val_split


def build_splits(val_ratio=0.25, seed=42):
    """
    Build all dataset splits (train/val/test) and their numeric versions.
    Safe for importing inside notebooks or other scripts.

    Returns a single dictionary:
    {
        "train": {...},
        "train_numeric": {...},
        "val": {...},
        "val_numeric": {...},
        "test": {...},
        "test_numeric": {...}
    }
    """

    # Load environment variables
    BASE_DIR = Path(__file__).resolve().parents[2]  # src/scripts → repo root
    ENV_PATH = BASE_DIR / ".env"
    load_dotenv(ENV_PATH)

    DS_PATH = os.getenv("FER2013_PATH")
    if DS_PATH is None:
        raise ValueError("FER2013_PATH not found in .env")

    # Root folders
    train_root = os.path.join(DS_PATH, "train")
    test_root = os.path.join(DS_PATH, "test")

    # Build dicts
    full_train_dict = build_dict(train_root)
    train_dict, val_dict = split_train_val_sklearn(full_train_dict, val_ratio, seed)
    test_dict = build_dict(test_root)

    # Numeric versions
    numeric_train_dict = build_numeric_dict(train_dict)
    numeric_val_dict = build_numeric_dict(val_dict)
    numeric_test_dict = build_numeric_dict(test_dict)

    return {
        "train": train_dict,
        "train_numeric": numeric_train_dict,
        "val": val_dict,
        "val_numeric": numeric_val_dict,
        "test": test_dict,
        "test_numeric": numeric_test_dict
    }



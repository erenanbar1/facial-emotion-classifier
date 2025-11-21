import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np

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


def main():
    # Load environment variables
    load_dotenv("../../.env")
    DS_PATH = os.getenv("FER2013_PATH")

    if DS_PATH is None:
        raise ValueError("FER2013_PATH not found in .env")

    # Root folders
    train_root = os.path.join(DS_PATH, "train")
    test_root = os.path.join(DS_PATH, "test")

    # Build dictionaries
    train_dict = build_dict(train_root)
    test_dict = build_dict(test_root)

    numeric_train_dict = build_numeric_dict(train_dict)
    numeric_test_dict = build_numeric_dict(test_dict)

    # Optional: print summary
    print("Train classes:", list(train_dict.keys()))
    print("Test classes: ", list(test_dict.keys()))
    for cls in train_dict:
        print(f"{cls}: {len(train_dict[cls])} train images, {len(test_dict[cls])} test images")


    # Return values if needed by other scripts
    return train_dict, numeric_train_dict, test_dict, numeric_test_dict


if __name__ == "__main__":
    main()

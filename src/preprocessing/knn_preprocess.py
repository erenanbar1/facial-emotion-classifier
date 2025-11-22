import numpy as np
import random
from typing import Dict, List
from PIL import Image, ImageOps, ImageEnhance


Array = np.ndarray
SplitDict = Dict[str, List[Array]]


# -----------------------------------------------------------
# 1) simple augmentations for KNN (PIL → numpy)
# -----------------------------------------------------------
def augment_single(img_arr: Array) -> Array:
    """
    For KNN, keep augmentations small. Convert numpy → PIL → numpy.
    img_arr is float32 in [0,1], shape (H,W).
    """
    pil = Image.fromarray((img_arr * 255).astype("uint8"), mode="L")

    # Random horizontal flip
    if random.random() < 0.5:
        pil = ImageOps.mirror(pil)

    # Small rotation ±10 degrees
    pil = pil.rotate(random.uniform(-10, 10), resample=Image.BILINEAR)

    # Brightness/contrast jitter
    if random.random() < 0.5:
        pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.8, 1.2))

    arr = np.array(pil, dtype="float32") / 255.0
    return arr


# -----------------------------------------------------------
# 2) Up-sample a minority class for KNN
# -----------------------------------------------------------
def augment_class_to_target(images: List[Array], target_count: int) -> List[Array]:
    if len(images) >= target_count:
        return images

    output = list(images)
    while len(output) < target_count:
        base = random.choice(images)
        aug = augment_single(base)
        output.append(aug)
    return output


# -----------------------------------------------------------
# 3) MAIN FUNCTION FOR KNN PREPROCESSING
# -----------------------------------------------------------
def populate_skew_for_knn(train_dict: SplitDict, target_label: str, seed=42):
    """
    Balances the target_class in numeric_train_dict, but for KNN.
    Returns a *new* train_dict with augmented images.
    """

    random.seed(seed)

    # find largest class size
    max_size = max(len(v) for v in train_dict.values())

    # copy original
    new_dict = {cls: list(imgs) for cls, imgs in train_dict.items()}

    # apply augmentation only to target class
    new_dict[target_label] = augment_class_to_target(
        new_dict[target_label],
        target_count=max_size
    )

    return new_dict


# -----------------------------------------------------------
# 4) Convert class dict → X, y matrices for KNN
# -----------------------------------------------------------
def dict_to_xy(train_dict: SplitDict):
    X = []
    y = []

    for cls, imgs in train_dict.items():
        for img in imgs:
            X.append(img.flatten())  # shape → (2304,)
            y.append(cls)

    X = np.array(X, dtype="float32")
    y = np.array(y)

    return X, y

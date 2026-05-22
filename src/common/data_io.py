import os
from struct import unpack
import numpy as np


def load_mnist_idx(path, kind="train"):
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte")

    with open(labels_path, "rb") as lbpath:
        _, _ = unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        _, _, rows, cols = unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), rows, cols)

    return images, labels

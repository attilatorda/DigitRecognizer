# Reference MNIST accuracy figures for positioning the one-shot results.
# Accuracies are on the standard 10K test set unless noted.

BENCHMARKS = [
    {
        "label": "Random chance (10-class)",
        "accuracy": 0.10,
        "source": "theoretical",
    },
    {
        "label": "Nearest-neighbour (L2, raw pixels)",
        "accuracy": 0.9246,  # LeCun et al. 1998 — 1-NN on raw 28x28
        "source": "LeCun et al. (1998) Gradient-based learning applied to document recognition",
    },
    {
        "label": "Linear classifier (SVM, RBF)",
        "accuracy": 0.9840,
        "source": "LeCun et al. (1998)",
    },
    {
        "label": "LeNet-5 (LeCun 1998)",
        "accuracy": 0.9905,
        "source": "LeCun et al. (1998)",
    },
    {
        "label": "SimpleCNN (this project, full supervision)",
        "accuracy": None,  # filled at runtime from local_cnn results
        "source": "this project — src/local_cnn/train_local_cnn.py",
    },
    {
        "label": "Human error estimate",
        "accuracy": 0.9977,
        "source": "Simard et al. (2003) Best practices for CNNs",
    },
]

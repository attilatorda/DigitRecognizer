LABELS_17 = [
    "4_variant_a",
    "0_variant_a",
    "1_variant_a",
    "2_variant_a",
    "4_variant_b",
    "7_variant_a",
    "9_variant_a",
    "0_variant_b",
    "1_variant_b",
    "2_variant_b",
    "3",
    "4_variant_c",
    "5",
    "6",
    "7_variant_b",
    "8",
    "9_variant_b",
]

LABEL_TO_ID_17 = {name: idx for idx, name in enumerate(LABELS_17)}
ID_TO_LABEL_17 = {idx: name for name, idx in LABEL_TO_ID_17.items()}

# Map variant class id -> canonical MNIST digit id for evaluation on MNIST.
CLASS17_TO_DIGIT10 = {
    LABEL_TO_ID_17["4_variant_a"]: 4,
    LABEL_TO_ID_17["0_variant_a"]: 0,
    LABEL_TO_ID_17["1_variant_a"]: 1,
    LABEL_TO_ID_17["2_variant_a"]: 2,
    LABEL_TO_ID_17["4_variant_b"]: 4,
    LABEL_TO_ID_17["7_variant_a"]: 7,
    LABEL_TO_ID_17["9_variant_a"]: 9,
    LABEL_TO_ID_17["0_variant_b"]: 0,
    LABEL_TO_ID_17["1_variant_b"]: 1,
    LABEL_TO_ID_17["2_variant_b"]: 2,
    LABEL_TO_ID_17["3"]: 3,
    LABEL_TO_ID_17["4_variant_c"]: 4,
    LABEL_TO_ID_17["5"]: 5,
    LABEL_TO_ID_17["6"]: 6,
    LABEL_TO_ID_17["7_variant_b"]: 7,
    LABEL_TO_ID_17["8"]: 8,
    LABEL_TO_ID_17["9_variant_b"]: 9,
}

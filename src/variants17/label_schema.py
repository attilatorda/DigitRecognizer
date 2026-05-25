LABELS_17 = [
    "open_four",
    "crossed_zero",
    "european_one",
    "two",
    "four",
    "european_seven",
    "nine_variant_a",
    "zero",
    "american_one",
    "curly_two",
    "three",
    "irregular_four",
    "five",
    "six",
    "american_seven",
    "eight",
    "nine_variant_b",
]

LABEL_TO_ID_17 = {name: idx for idx, name in enumerate(LABELS_17)}
ID_TO_LABEL_17 = {idx: name for name, idx in LABEL_TO_ID_17.items()}

# Map variant class id -> canonical MNIST digit id for evaluation on MNIST.
CLASS17_TO_DIGIT10 = {
    LABEL_TO_ID_17["open_four"]: 4,
    LABEL_TO_ID_17["crossed_zero"]: 0,
    LABEL_TO_ID_17["european_one"]: 1,
    LABEL_TO_ID_17["two"]: 2,
    LABEL_TO_ID_17["four"]: 4,
    LABEL_TO_ID_17["european_seven"]: 7,
    LABEL_TO_ID_17["nine_variant_a"]: 9,
    LABEL_TO_ID_17["zero"]: 0,
    LABEL_TO_ID_17["american_one"]: 1,
    LABEL_TO_ID_17["curly_two"]: 2,
    LABEL_TO_ID_17["three"]: 3,
    LABEL_TO_ID_17["irregular_four"]: 4,
    LABEL_TO_ID_17["five"]: 5,
    LABEL_TO_ID_17["six"]: 6,
    LABEL_TO_ID_17["american_seven"]: 7,
    LABEL_TO_ID_17["eight"]: 8,
    LABEL_TO_ID_17["nine_variant_b"]: 9,
}

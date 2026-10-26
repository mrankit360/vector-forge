import numpy as np


def euclidean(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na < 1e-9 or nb < 1e-9:
        return 1.0

    similarity = float(np.dot(a, b) / (na * nb))
    return 1.0 - max(-1.0, min(1.0, similarity))


def manhattan(a, b):
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))


def get_distance(metric):
    metric = metric.lower().strip()

    aliases = {
        "cos": "cosine",
        "l2": "euclidean",
        "l1": "manhattan",
        "l2sq": "euclidean",
    }
    metric = aliases.get(metric, metric)

    functions = {
        "euclidean": euclidean,
        "cosine": cosine,
        "manhattan": manhattan,
    }

    if metric not in functions:
        raise ValueError(f"Unsupported metric: {metric}")

    return functions[metric]
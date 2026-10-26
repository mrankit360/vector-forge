
import threading

import numpy as np
from sklearn.neighbors import KDTree

from .metrics import get_distance


class KDTreeIndex:
    def __init__(self, dims):
        if dims <= 0:
            raise ValueError("dims must be positive")

        self.dims = dims
        self.items = {}
        self.tree = None
        self.ids = []
        self.lock = threading.RLock()

    def _vector(self, embedding):
        v = np.asarray(embedding, dtype=np.float32)

        if v.shape != (self.dims,):
            raise ValueError(f"Expected ({self.dims},), got {v.shape}")

        if not np.all(np.isfinite(v)):
            raise ValueError("Embedding contains NaN or infinity")

        return np.ascontiguousarray(v)

    def _rebuild(self):
        self.ids = list(self.items)

        if not self.ids:
            self.tree = None
            return

        matrix = np.stack([
            self.items[item_id]["embedding"]
            for item_id in self.ids
        ])

        self.tree = KDTree(matrix, leaf_size=40)

    def insert(self, item):
        item_id = int(item["id"])
        v = self._vector(item["embedding"])

        with self.lock:
            if item_id in self.items:
                raise ValueError(f"Duplicate ID: {item_id}")

            stored = dict(item)
            stored["id"] = item_id
            stored["embedding"] = v.copy()
            self.items[item_id] = stored
            self._rebuild()

    def remove(self, item_id):
        with self.lock:
            if item_id not in self.items:
                return False

            del self.items[item_id]
            self._rebuild()
            return True

    def knn(self, query, k=5, metric="cosine"):
        q = self._vector(query)

        if k <= 0:
            return []

        metric = metric.lower().strip()
        metric = {
            "cos": "cosine",
            "l2": "euclidean",
            "l1": "manhattan",
            "l2sq": "euclidean",
        }.get(metric, metric)

        if metric not in ("cosine", "euclidean", "manhattan"):
            raise ValueError(f"Unsupported metric: {metric}")

        with self.lock:
            if not self.items:
                return []

            count = min(int(k), len(self.items))

            if metric == "cosine":
                fn = get_distance(metric)
                result = [
                    (fn(q, item["embedding"]), item_id)
                    for item_id, item in self.items.items()
                ]
                return sorted(result)[:count]

            self._rebuild()

            distances, indices = self.tree.query(
                q.reshape(1, -1),
                k=count,
                return_distance=True,
                dualtree=False,
            ) if self.tree is not None else ([], [])

            if self.tree is None:
                return []

            # KDTree's supported metric must match the requested metric.
            # Rebuild with the right metric when needed.
            if getattr(self.tree, "metric", None) != metric:
                matrix = np.stack([
                    self.items[item_id]["embedding"]
                    for item_id in self.ids
                ])
                self.tree = KDTree(matrix, leaf_size=40, metric=metric)
                distances, indices = self.tree.query(
                    q.reshape(1, -1), k=count
                )

            return sorted([
                (float(distances[0][j]), self.ids[int(indices[0][j])])
                for j in range(count)
            ])


import threading
from typing import Any

import numpy as np
from usearch.index import Index

from .metrics import get_distance


class HNSWIndex:
    """
    HNSW nearest-neighbor index backed by USearch.

    Stored items use this format:
        {
            "id": 1,
            "metadata": "Linked List",
            "category": "cs",
            "embedding": [...]
        }

    Returns search results as:
        [(distance, item_id), ...]
    """

    def __init__(
        self,
        dims: int,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        metric: str = "cosine",
    ):
        if dims <= 0:
            raise ValueError("dims must be positive")

        if m <= 0 or ef_construction <= 0 or ef_search <= 0:
            raise ValueError("HNSW parameters must be positive")

        self.dims = dims
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.metric = self._normalize_metric(metric)

        self.items: dict[int, dict[str, Any]] = {}
        self.index = None
        self.lock = threading.RLock()

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_metric(metric: str) -> str:
        metric = metric.lower().strip()

        aliases = {
            "cos": "cosine",
            "l2": "euclidean",
            "l2sq": "euclidean",
            "l1": "manhattan",
        }

        metric = aliases.get(metric, metric)

        if metric not in ("cosine", "euclidean", "manhattan"):
            raise ValueError(f"Unsupported metric: {metric}")

        return metric

    def _vector(self, embedding) -> np.ndarray:
        try:
            vector = np.asarray(embedding, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid embedding") from exc

        if vector.shape != (self.dims,):
            raise ValueError(
                f"Expected embedding shape ({self.dims},), "
                f"got {vector.shape}"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError("Embedding contains NaN or infinity")

        return np.ascontiguousarray(vector)

    def _validate_cosine(self, vector: np.ndarray) -> None:
        if self.metric == "cosine" and np.linalg.norm(vector) <= 1e-12:
            raise ValueError(
                "Zero vectors cannot be used for cosine search"
            )

    # ---------------------------------------------------------
    # INDEX CREATION
    # ---------------------------------------------------------

    def _ensure_index(self) -> None:
        if self.index is not None:
            return

        # USearch supports cosine, L2sq and inner-product metrics.
        # Manhattan is handled by exact search instead.
        if self.metric == "manhattan":
            return

        usearch_metric = {
            "cosine": "cos",
            "euclidean": "l2sq",
        }[self.metric]

        self.index = Index(
            ndim=self.dims,
            metric=usearch_metric,
            connectivity=self.m,
            expansion_add=self.ef_construction,
            expansion_search=self.ef_search,
        )

    # ---------------------------------------------------------
    # INSERT
    # ---------------------------------------------------------

    def insert(self, item: dict[str, Any]) -> None:
        if "id" not in item or "embedding" not in item:
            raise ValueError("Item must contain id and embedding")

        item_id = int(item["id"])
        vector = self._vector(item["embedding"])
        self._validate_cosine(vector)

        with self.lock:
            if item_id in self.items:
                raise ValueError(f"Duplicate ID: {item_id}")

            self._ensure_index()

            # Keep the Python metadata and native index consistent.
            if self.index is not None:
                self.index.add(item_id, vector)

            stored_item = item.copy()
            stored_item["id"] = item_id
            stored_item["embedding"] = vector.copy()

            self.items[item_id] = stored_item

    # ---------------------------------------------------------
    # DELETE
    # ---------------------------------------------------------

    def remove(self, item_id: int) -> bool:
        item_id = int(item_id)

        with self.lock:
            if item_id not in self.items:
                return False

            if self.index is not None:
                self.index.remove(item_id)

            del self.items[item_id]
            return True

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------

    def knn(
        self,
        query,
        k: int = 5,
        metric: str | None = None,
        exact: bool = False,
    ) -> list[tuple[float, int]]:
        query = self._vector(query)

        if k <= 0:
            return []

        selected_metric = self._normalize_metric(
            metric if metric is not None else self.metric
        )

        if selected_metric == "cosine":
            if np.linalg.norm(query) <= 1e-12:
                raise ValueError(
                    "Zero query vector cannot be used for cosine search"
                )

        with self.lock:
            if not self.items:
                return []

            count = min(int(k), len(self.items))

            # Exact search is necessary when the requested metric
            # differs from the metric used to construct the index.
            if exact or selected_metric != self.metric:
                return self._exact_search(query, count, selected_metric)

            # Manhattan is not handled by the approximate index.
            if selected_metric == "manhattan":
                return self._exact_search(query, count, selected_metric)

            self._ensure_index()

            if self.index is None:
                return self._exact_search(query, count, selected_metric)

            self.index.expansion_search = max(self.ef_search, count)

            matches = self.index.search(query, count)

            results = []

            for match in matches:
                item_id = int(match.key)

                if item_id not in self.items:
                    continue

                # USearch's L2 metric returns squared Euclidean
                # distance. Convert it to ordinary Euclidean distance.
                distance = float(match.distance)

                if selected_metric == "euclidean":
                    distance = float(np.sqrt(max(0.0, distance)))

                if np.isfinite(distance):
                    results.append((distance, item_id))

            results.sort(key=lambda result: (result[0], result[1]))
            return results[:count]

    # ---------------------------------------------------------
    # EXACT SEARCH
    # ---------------------------------------------------------

    def _exact_search(
        self,
        query: np.ndarray,
        k: int,
        metric: str,
    ) -> list[tuple[float, int]]:
        distance_fn = get_distance(metric)
        results = []

        for item_id, item in self.items.items():
            vector = item["embedding"]
            distance = float(distance_fn(query, vector))

            if np.isfinite(distance):
                results.append((distance, item_id))

        results.sort(key=lambda result: (result[0], result[1]))
        return results[:k]

    # ---------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------

    def get(self, item_id: int):
        with self.lock:
            item = self.items.get(int(item_id))

            if item is None:
                return None

            result = item.copy()
            result["embedding"] = item["embedding"].copy()
            return result

    def all(self) -> list[dict[str, Any]]:
        with self.lock:
            results = []

            for item in self.items.values():
                result = item.copy()
                result["embedding"] = item["embedding"].copy()
                results.append(result)

            return results

    def size(self) -> int:
        with self.lock:
            return len(self.items)

    def clear(self) -> None:
        with self.lock:
            self.items.clear()
            self.index = None

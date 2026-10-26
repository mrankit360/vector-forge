
import threading

import numpy as np

from .metrics import cosine
from .hnsw_index import HNSWIndex


class DocumentDB:
    def __init__(self):
        self.store = {}
        self.hnsw = None
        self.dims = 0
        self.next_id = 1
        self.lock = threading.RLock()

    def insert(self, title, text, embedding):
        vector = np.asarray(embedding, dtype=np.float32)

        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("Embedding must be a non-empty vector")

        if not np.all(np.isfinite(vector)):
            raise ValueError("Embedding contains NaN or infinity")

        if np.linalg.norm(vector) <= 1e-12:
            raise ValueError("Zero vectors cannot use cosine search")

        with self.lock:
            if self.dims == 0:
                self.dims = len(vector)
                self.hnsw = HNSWIndex(
                    self.dims, m=16,
                    ef_construction=200, ef_search=50
                )

            if len(vector) != self.dims:
                raise ValueError(
                    f"Expected {self.dims} dimensions, got {len(vector)}"
                )

            item_id = self.next_id
            item = {
                "id": item_id,
                "title": title,
                "text": text,
                "embedding": vector.copy(),
            }

            self.hnsw.insert({
                "id": item_id,
                "metadata": title,
                "category": "doc",
                "embedding": vector,
            })

            self.store[item_id] = item
            self.next_id += 1
            return item_id

    def search(self, query, k=3, max_dist=0.7):
        query = np.asarray(query, dtype=np.float32)

        with self.lock:
            if not self.store or k <= 0:
                return []

            if query.shape != (self.dims,):
                raise ValueError(f"Expected a {self.dims}D query")

            if not np.all(np.isfinite(query)):
                raise ValueError("Query contains NaN or infinity")

            if np.linalg.norm(query) <= 1e-12:
                raise ValueError("Zero query vector cannot use cosine search")

            if len(self.store) < 10:
                raw = [
                    (cosine(query, item["embedding"]), item_id)
                    for item_id, item in self.store.items()
                ]
                raw.sort()
                raw = raw[:k]
            else:
                raw = self.hnsw.knn(query, k=k, metric="cosine")

            return [
                (distance, self.store[item_id].copy())
                for distance, item_id in raw
                if item_id in self.store and distance <= max_dist
            ]

    def remove(self, item_id):
        with self.lock:
            if item_id not in self.store:
                return False

            self.hnsw.remove(item_id)
            del self.store[item_id]
            return True

    def all(self):
        with self.lock:
            return [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "text": item["text"],
                    "embedding": item["embedding"].tolist(),
                }
                for item in self.store.values()
            ]

    def size(self):
        with self.lock:
            return len(self.store)

    def get_dims(self):
        with self.lock:
            return self.dims

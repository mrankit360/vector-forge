
import threading
import time

import numpy as np

from .metrics import get_distance
from .hnsw_index import HNSWIndex
from .kd_tree import KDTreeIndex


class BruteForceIndex:
    def __init__(self, dims):
        self.dims = dims
        self.items = {}

    def insert(self, item):
        self.items[item["id"]] = item

    def remove(self, item_id):
        self.items.pop(item_id, None)

    def knn(self, query, k=5, metric="cosine"):
        if k <= 0:
            return []

        fn = get_distance(metric)
        results = [
            (fn(query, item["embedding"]), item_id)
            for item_id, item in self.items.items()
        ]
        return sorted(results)[:k]


class VectorDB:
    def __init__(self, dims=16):
        self.dims = dims
        self.store = {}
        self.bf = BruteForceIndex(dims)
        self.kdt = KDTreeIndex(dims)
        self.hnsw = HNSWIndex(dims, m=16, ef_construction=200, ef_search=50)
        self.next_id = 1
        self.lock = threading.RLock()

    def insert(self, metadata, category, embedding):
        vector = np.asarray(embedding, dtype=np.float32)

        if vector.shape != (self.dims,):
            raise ValueError(f"Expected {self.dims} dimensions")

        if not np.all(np.isfinite(vector)):
            raise ValueError("Embedding contains NaN or infinity")

        with self.lock:
            item_id = self.next_id
            item = {
                "id": item_id,
                "metadata": metadata,
                "category": category,
                "embedding": vector.copy(),
            }

            # Insert into each index. Roll back already-inserted indexes
            # if a later index operation fails.
            inserted = []
            try:
                self.bf.insert(item)
                inserted.append(self.bf)

                self.kdt.insert(item)
                inserted.append(self.kdt)

                self.hnsw.insert(item)
                inserted.append(self.hnsw)
            except Exception:
                for index in reversed(inserted):
                    index.remove(item_id)
                raise

            self.store[item_id] = item
            self.next_id += 1
            return item_id

    def remove(self, item_id):
        item_id = int(item_id)

        with self.lock:
            if item_id not in self.store:
                return False

            # If a library operation fails, report the failure rather
            # than silently claiming the item was deleted.
            self.bf.remove(item_id)
            self.kdt.remove(item_id)
            self.hnsw.remove(item_id)
            del self.store[item_id]
            return True

    def search(self, query, k=5, metric="cosine", algo="hnsw"):
        query = np.asarray(query, dtype=np.float32)

        if query.shape != (self.dims,):
            raise ValueError(f"Expected a {self.dims}D query")

        if not np.all(np.isfinite(query)):
            raise ValueError("Query contains NaN or infinity")

        algo = algo.lower().strip()
        metric = metric.lower().strip()

        with self.lock:
            started = time.perf_counter()

            if algo == "bruteforce":
                raw = self.bf.knn(query, k, metric)
            elif algo == "kdtree":
                raw = self.kdt.knn(query, k, metric)
            elif algo == "hnsw":
                raw = self.hnsw.knn(query, k, metric)
            else:
                raise ValueError(f"Unsupported algorithm: {algo}")

            elapsed_us = int((time.perf_counter() - started) * 1_000_000)

            hits = []
            for distance, item_id in raw:
                if item_id not in self.store:
                    continue

                item = self.store[item_id]
                hits.append({
                    "id": item_id,
                    "metadata": item["metadata"],
                    "category": item["category"],
                    "embedding": item["embedding"].tolist(),
                    "distance": float(distance),
                })

            return {
                "results": hits,
                "latencyUs": elapsed_us,
                "algo": algo,
                "metric": metric,
            }

    def benchmark(self, query, k=5, metric="cosine"):
        return {
            "bruteforceUs": self.search(
                query, k, metric, "bruteforce"
            )["latencyUs"],
            "kdtreeUs": self.search(
                query, k, metric, "kdtree"
            )["latencyUs"],
            "hnswUs": self.search(
                query, k, metric, "hnsw"
            )["latencyUs"],
            "itemCount": self.size(),
        }

    def all(self):
        with self.lock:
            return [
                {
                    **item,
                    "embedding": item["embedding"].tolist(),
                }
                for item in self.store.values()
            ]

    def size(self):
        with self.lock:
            return len(self.store)

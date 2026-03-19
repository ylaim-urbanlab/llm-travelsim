"""
LLM response cache with two modes:
  - "fingerprint": key on bucketed semantic context (default) — reuses trips across
                   similar hexes/times without identical prompts
  - "prompt":      key on exact prompt hash — perfect fidelity, only reuses on
                   identical inputs
"""
from __future__ import annotations
import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional


# How many buckets to quantize each continuous feature into for fingerprinting.
# Coarser = more cache hits, less variety. 5 is a reasonable default.
N_BUCKETS = 5

# Population thresholds for bucketing (people per hex)
POP_BUCKETS = [100, 300, 600, 1000, 2000]


def _bucket(value: float, n: int = N_BUCKETS) -> int:
    """Map a 0-1 float into one of n integer buckets."""
    return min(int(value * n), n - 1)


def _pop_bucket(pop: float) -> int:
    for i, threshold in enumerate(POP_BUCKETS):
        if pop < threshold:
            return i
    return len(POP_BUCKETS)


def _fingerprint(context: dict, cluster_memberships: List[float]) -> str:
    """
    Build a semantic fingerprint from bucketed context features.
    Two hexes with similar demographics, attractors, population, and time period
    will map to the same key and share cached trips.
    """
    demo = context.get("own_demographics", [])
    attr = context.get("own_attractors", [])
    period = context.get("period", "")
    pop = context.get("own_population", 0)
    dominant_cluster = int(max(range(len(cluster_memberships)),
                               key=lambda i: cluster_memberships[i]))

    parts = [
        period,
        f"c{dominant_cluster}",
        f"p{_pop_bucket(pop)}",
    ]
    parts += [f"d{i}={_bucket(v)}" for i, v in enumerate(demo)]
    parts += [f"a{i}={_bucket(v)}" for i, v in enumerate(attr)]

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


class LLMCache:
    """
    Thread-safe disk-backed cache for LLM trip responses.

    Args:
        cache_dir: directory to store the cache file
        mode: "fingerprint" (default) or "prompt"
    """

    def __init__(self, cache_dir: str, mode: str = "fingerprint"):
        if mode not in ("fingerprint", "prompt"):
            raise ValueError(f"Unknown cache mode: {mode!r}. Use 'fingerprint' or 'prompt'.")
        self.mode = mode
        self._path = Path(cache_dir) / f"llm_cache_{mode}.json"
        self._data: Dict[str, list] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self._hits = 0
        self._misses = 0
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
                print(f"[cache] Loaded {len(self._data):,} entries from {self._path} (mode={self.mode})")
            except Exception as e:
                print(f"[cache] Failed to load cache ({e}), starting fresh")
                self._data = {}

    def _make_key(self, prompt: str, context: dict, cluster_memberships: List[float]) -> str:
        if self.mode == "prompt":
            return _prompt_hash(prompt)
        else:
            return _fingerprint(context, cluster_memberships)

    def get(
        self,
        prompt: str,
        context: dict,
        cluster_memberships: List[float],
    ) -> Optional[list]:
        key = self._make_key(prompt, context, cluster_memberships)
        with self._lock:
            result = self._data.get(key)
            if result is not None:
                self._hits += 1
            else:
                self._misses += 1
            return result

    def put(
        self,
        prompt: str,
        context: dict,
        cluster_memberships: List[float],
        trips: list,
    ):
        key = self._make_key(prompt, context, cluster_memberships)
        with self._lock:
            self._data[key] = trips
            self._dirty = True

    def flush(self):
        """Write cache to disk. Call after each timestep."""
        with self._lock:
            if not self._dirty:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2))
            self._dirty = False

    @property
    def stats(self) -> str:
        total = self._hits + self._misses
        hit_rate = self._hits / total * 100 if total else 0
        return (f"cache hits: {self._hits}/{total} ({hit_rate:.0f}%) | "
                f"entries: {len(self._data):,} | mode: {self.mode}")

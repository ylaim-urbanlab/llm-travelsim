"""
Gaussian Mixture Model clustering of hexes by demographic features.
Returns soft cluster memberships, human-readable labels, and importance weights.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from llm_travel_sim.data.census import FEATURE_NAMES


# Interpretive label templates based on cluster centroid characteristics
def _label_cluster(centroid: np.ndarray, cluster_idx: int) -> str:
    """Generate a human-readable label from a cluster centroid."""
    (income, poverty, snap, age, pct_under18,
     family_hh, renter, zero_vehicle, wfh, commute,
     mgmt_prof, service) = centroid

    parts = []

    # Age description
    if age < 0.35:
        parts.append("young")
    elif age > 0.55:
        parts.append("older")

    # Income / poverty
    if income > 0.6 and poverty < 0.15:
        parts.append("affluent")
    elif poverty > 0.3 or snap > 0.25:
        parts.append("low-income")
    elif income < 0.35:
        parts.append("working-class")

    # Urban vs suburban
    if renter > 0.6 and zero_vehicle > 0.15:
        parts.append("urban renter")
    elif renter < 0.3 and family_hh > 0.6:
        parts.append("suburban homeowner")
    elif renter > 0.5:
        parts.append("renter")

    # Family structure
    if pct_under18 > 0.3:
        parts.append("with children")
    elif family_hh < 0.3:
        parts.append("single-person")

    # Mobility
    if zero_vehicle > 0.25:
        parts.append("car-free")
    elif zero_vehicle < 0.05:
        parts.append("car-dependent")

    # Work
    if wfh > 0.25:
        parts.append("remote worker")
    elif mgmt_prof > 0.5:
        parts.append("professional")
    elif service > 0.35:
        parts.append("service worker")

    if not parts:
        parts.append(f"mixed household type {cluster_idx + 1}")

    return " ".join(parts[:4])  # keep it concise


def cluster_cache_basename(hex_ids: List[str], n_clusters: int) -> str:
    """Return cache filename for clusters keyed by hex set + k (for inspect_hex / main)."""
    hex_set_hash = hashlib.md5("|".join(sorted(hex_ids)).encode()).hexdigest()[:8]
    return f"clusters_k{n_clusters}_{hex_set_hash}.json"


def fit_clusters(
    hex_features: Dict[str, np.ndarray],
    n_clusters: int,
    random_seed: int,
    cache_dir: str,
) -> Tuple[Dict[str, np.ndarray], List[str], np.ndarray]:
    """
    Fit GMM, return:
    - per-hex soft cluster membership vectors {hex_id: array shape (k,)}
    - cluster labels list of length k
    - importance weights array shape (k,) for stratified reweighting
    """
    cache_path = Path(cache_dir) / cluster_cache_basename(list(hex_features.keys()), n_clusters)
    if cache_path.exists():
        print(f"[clustering] Loading clusters from cache: {cache_path}")
        raw = json.loads(cache_path.read_text())
        memberships = {k: np.array(v, dtype=np.float32) for k, v in raw["memberships"].items()}
        labels = raw["labels"]
        weights = np.array(raw["weights"], dtype=np.float32)
        return memberships, labels, weights

    hex_ids = list(hex_features.keys())
    X = np.stack([hex_features[hid] for hid in hex_ids])
    n_clusters = min(n_clusters, len(hex_ids))
    print(f"[clustering] Fitting GMM with k={n_clusters} clusters...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_seed,
        max_iter=200,
        n_init=5,
    )
    gmm.fit(X_scaled)

    # Soft membership probabilities
    probs = gmm.predict_proba(X_scaled).astype(np.float32)  # shape (n_hexes, k)
    memberships = {hid: probs[i] for i, hid in enumerate(hex_ids)}

    # Cluster centroids (unscaled) for labeling
    centroids_scaled = gmm.means_  # shape (k, n_features)
    centroids = scaler.inverse_transform(centroids_scaled)
    centroids_clipped = np.clip(centroids, 0, 1)

    labels = [_label_cluster(centroids_clipped[i], i) for i in range(n_clusters)]
    print(f"[clustering] Cluster labels: {labels}")

    # Population shares
    cluster_assignments = np.argmax(probs, axis=1)
    cluster_counts = np.bincount(cluster_assignments, minlength=n_clusters).astype(float)
    pop_shares = cluster_counts / cluster_counts.sum()

    # Importance weights for stratified sampling: inverse frequency with floor
    min_share = 1.0 / (n_clusters * 3)
    effective_shares = np.maximum(pop_shares, min_share)
    weights = (1.0 / effective_shares)
    weights = (weights / weights.sum() * n_clusters).astype(np.float32)

    cache_path.write_text(json.dumps({
        "memberships": {k: v.tolist() for k, v in memberships.items()},
        "labels": labels,
        "weights": weights.tolist(),
    }, indent=2))
    print(f"[clustering] Clusters fitted and cached to {cache_path}")
    return memberships, labels, weights

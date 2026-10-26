#!/usr/bin/env python3
"""Utilities for dimension-aware DFTHC-SOS pruning and candidate selection.

The numerical trainer itself is supplied by the official Kaggle starter kit.
This module contains the deterministic parts of the v7 search: contribution
ranking, shape reduction, strict metric checks and incumbent comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


REQUIRED_PARAMETER_KEYS = ("U", "W", "H2_I", "H1_I", "Hsym_lower")


def numpy_parameters(params: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Copy the five submitted arrays into ordinary native NumPy arrays."""
    missing = [name for name in REQUIRED_PARAMETER_KEYS if name not in params]
    if missing:
        raise KeyError(f"missing factorization arrays: {missing}")
    result = {name: np.asarray(params[name]).copy() for name in REQUIRED_PARAMETER_KEYS}
    if result["U"].ndim != 3 or result["W"].ndim != 3 or result["H2_I"].ndim != 2:
        raise ValueError("expected U(R,B,N), W(R,B,C), and H2_I(R,C)")
    r, b, _ = result["U"].shape
    if result["W"].shape[:2] != (r, b):
        raise ValueError("U and W disagree on (R,B)")
    if result["H2_I"].shape != (r, result["W"].shape[2]):
        raise ValueError("H2_I and W disagree on (R,C)")
    return result


def _descending_indices(score: np.ndarray, count: int, axis: int) -> np.ndarray:
    if not 0 < count <= score.shape[axis]:
        raise ValueError(f"cannot retain {count} entries along shape {score.shape}")
    # Stable sorting makes checkpoint production deterministic when scores tie.
    return np.flip(np.argsort(score, axis=axis, kind="stable"), axis=axis).take(
        indices=range(count), axis=axis
    )


def prune_dimensions(
    params: Mapping[str, Any],
    rank_r: int,
    bases_b: int,
    ensemble_c: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reduce an existing factorization using contribution-ranked slices.

    Groups are ranked globally.  Basis and ensemble coordinates are ranked
    independently inside every retained group.  Reordering is legal because
    those axes are summed by the verifier.  Training must still repair the
    numerical approximation; this function makes no acceptance claim.
    """
    source = numpy_parameters(params)
    u, w, h2 = source["U"], source["W"], source["H2_I"]
    r0, b0, c0 = w.shape
    if rank_r > r0 or bases_b > b0 or ensemble_c > c0:
        raise ValueError(
            f"target {(rank_r, bases_b, ensemble_c)} must not exceed source {(r0, b0, c0)}"
        )

    # U vectors are close to unit length, but include their measured norm so
    # this also behaves sensibly on intermediate, unnormalised checkpoints.
    # The verifier stores H2_I but uses the identity offset
    # W_offset = H2_I - sum_b(W).  Preserve that offset exactly while basis
    # entries are removed; keeping H2_I itself would inject the removed basis
    # weights into every retained offset and gives the trainer a poor start.
    offset = h2 - np.sum(w, axis=1)
    u_norm = np.linalg.norm(u, axis=2)
    basis_score = np.sum(np.abs(w), axis=2) * np.maximum(u_norm, 1e-12)
    group_score = np.sum(basis_score, axis=1) + np.sum(np.abs(offset), axis=1)
    group_index = _descending_indices(group_score, rank_r, axis=0)

    u = u[group_index]
    w = w[group_index]
    offset = offset[group_index]
    basis_score = basis_score[group_index]
    basis_index = _descending_indices(basis_score, bases_b, axis=1)
    u = np.take_along_axis(u, basis_index[:, :, None], axis=1)
    w = np.take_along_axis(w, basis_index[:, :, None], axis=1)

    ensemble_score = np.sum(np.abs(w), axis=1) + np.abs(offset)
    ensemble_index = _descending_indices(ensemble_score, ensemble_c, axis=1)
    w = np.take_along_axis(w, ensemble_index[:, None, :], axis=2)
    offset = np.take_along_axis(offset, ensemble_index, axis=1)
    h2 = offset + np.sum(w, axis=1)

    reduced = {
        "U": np.ascontiguousarray(u),
        "W": np.ascontiguousarray(w),
        "H2_I": np.ascontiguousarray(h2),
        "H1_I": source["H1_I"],
        "Hsym_lower": source["Hsym_lower"],
    }
    record = {
        "source_shape": [r0, b0, c0],
        "target_shape": [rank_r, bases_b, ensemble_c],
        "group_index": group_index.tolist(),
        "basis_index": basis_index.tolist(),
        "ensemble_index": ensemble_index.tolist(),
        "retained_group_score_fraction": float(
            np.sum(group_score[group_index]) / max(np.sum(group_score), 1e-300)
        ),
    }
    return reduced, record


def accepted_metrics(metrics: Mapping[str, Any]) -> bool:
    """Mirror the verifier's strict numerical acceptance predicates."""
    return bool(
        float(metrics["error_1"]) <= 0.020
        and float(metrics["error_2"]) <= 0.020
        and float(metrics["Lambda"]) > 49.0
        and float(metrics["E_gap"]) > 3.914898
        and bool(metrics["gap_in_domain"])
        and math.isfinite(float(metrics["lambda_eff"]))
    )


@dataclass(frozen=True)
class CandidateCost:
    shape: tuple[int, int, int]
    lambda_eff: float
    toffoli: float
    qubits: int
    cost: float
    accepted: bool

    @property
    def beats_incumbent(self) -> bool:
        return self.accepted and math.isfinite(self.cost)


def candidate_cost(
    shape: tuple[int, int, int],
    metrics: Mapping[str, Any],
    resources: Mapping[str, Any],
    *,
    incumbent: float | None = None,
) -> CandidateCost:
    accepted = accepted_metrics(metrics) and int(resources["angles_qubits"]) <= 128
    value = (
        float(metrics["lambda_eff"])
        * float(resources["toffoli"])
        * int(resources["qubits"])
        if accepted
        else math.inf
    )
    if incumbent is not None and not value < incumbent:
        accepted = False
    return CandidateCost(
        shape=tuple(map(int, shape)),
        lambda_eff=float(metrics["lambda_eff"]),
        toffoli=float(resources["toffoli"]),
        qubits=int(resources["qubits"]),
        cost=float(value),
        accepted=accepted,
    )


def self_test() -> None:
    rng = np.random.default_rng(20261026)
    params = {
        "U": rng.normal(size=(5, 7, 11)),
        "W": rng.normal(size=(5, 7, 6)),
        "H2_I": rng.normal(size=(5, 6)),
        "H1_I": rng.normal(size=(11, 11)),
        "Hsym_lower": rng.normal(size=(11, 11)),
    }
    a, record = prune_dimensions(params, 4, 5, 3)
    b, record2 = prune_dimensions(params, 4, 5, 3)
    assert a["U"].shape == (4, 5, 11)
    assert a["W"].shape == (4, 5, 3)
    assert a["H2_I"].shape == (4, 3)
    assert all(np.array_equal(a[name], b[name]) for name in REQUIRED_PARAMETER_KEYS)
    assert record == record2
    original_offset = params["H2_I"] - np.sum(params["W"], axis=1)
    expected_offset = np.take_along_axis(
        original_offset[np.asarray(record["group_index"])],
        np.asarray(record["ensemble_index"]),
        axis=1,
    )
    actual_offset = a["H2_I"] - np.sum(a["W"], axis=1)
    assert np.allclose(actual_offset, expected_offset)


if __name__ == "__main__":
    self_test()
    print("dimension-pruning self-test: PASS")

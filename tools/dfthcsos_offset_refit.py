#!/usr/bin/env python3
"""Verifier-aligned fixed-word offset refit for a DFTHC-SOS submission.

The ANGLE circuits and all ``W`` tensor entries stay fixed.  Only the stored
``H2_I`` array is changed, which changes the verifier's offset
``W_offset = H2_I - sum_b(W)``.  Candidate selection always uses the official
``evaluate_words`` and ``passes_accuracy`` functions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import minimize


@dataclass
class OffsetModel:
    """Fast lambda objective and analytic gradient at fixed decoded vectors."""

    h0: np.ndarray
    x: np.ndarray
    w: np.ndarray
    beta_1: float
    ecore: float
    electrons: int
    energy_upper: float
    start: np.ndarray

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.w = np.asarray(self.w, dtype=np.float64)
        self.start = np.asarray(self.start, dtype=np.float64)
        self.sum_w = np.sum(self.w, axis=1)
        self.l1_w = np.linalg.norm(self.w, ord=1, axis=1)
        self.trace_x = np.trace(self.x, axis1=-2, axis2=-1)

    def value_gradient(
        self,
        flat: np.ndarray,
        *,
        gap_floor: float = 3.914899,
        gap_margin: float = 2e-6,
        floor_penalty: float = 2e6,
        trust_penalty: float = 1e-7,
    ) -> tuple[float, np.ndarray]:
        offsets = np.asarray(flat, dtype=np.float64).reshape(self.start.shape)
        h1_prime = self.h0 - np.einsum("rc,rcpq->pq", offsets, self.x, optimize=True)
        eigenvalues, eigenvectors = np.linalg.eigh(h1_prime)
        signs = np.sign(eigenvalues)
        sign_h = (eigenvectors * signs[None, :]) @ eigenvectors.T
        overlap = np.einsum("pq,rcpq->rc", sign_h, self.x, optimize=True)

        lambda_h1 = float(np.sum(np.abs(eigenvalues)))
        w_b = offsets + self.sum_w
        widths = self.l1_w + np.abs(w_b)
        lambda_h2 = 0.25 * float(np.sum(widths * widths))
        Lambda = lambda_h1 + lambda_h2

        trace_h1 = float(np.sum(eigenvalues))
        e_sos = (
            self.ecore
            - self.electrons * self.beta_1
            - (lambda_h1 - trace_h1)
            - 0.5 * float(np.sum(offsets * offsets))
        )
        e_gap = self.energy_upper - e_sos
        radicand = e_gap * (2.0 * Lambda - e_gap)
        if not np.isfinite(radicand) or radicand <= 0:
            # A large smooth-ish rejection value keeps L-BFGS-B inside domain.
            delta = offsets - self.start
            return 1e9 + float(np.sum(delta * delta)), (2.0 * delta).ravel()

        lambda_eff = math.sqrt(radicand)
        grad_lambda_h1 = -overlap
        grad_lambda_h2 = 0.5 * widths * np.sign(w_b)
        grad_Lambda = grad_lambda_h1 + grad_lambda_h2
        grad_gap = -overlap + self.trace_x + offsets
        gradient = (
            (Lambda - e_gap) * grad_gap + e_gap * grad_Lambda
        ) / lambda_eff

        objective = lambda_eff
        minimum_gap = gap_floor + gap_margin
        if e_gap < minimum_gap:
            shortfall = minimum_gap - e_gap
            objective += floor_penalty * shortfall * shortfall
            gradient += -2.0 * floor_penalty * shortfall * grad_gap
        if Lambda <= 49.000001:
            shortfall = 49.000001 - Lambda
            objective += floor_penalty * shortfall * shortfall
            gradient += -2.0 * floor_penalty * shortfall * grad_Lambda

        delta = offsets - self.start
        objective += trust_penalty * float(np.sum(delta * delta))
        gradient += 2.0 * trust_penalty * delta
        return float(objective), gradient.ravel()


def _build_model(inputs: Any, words: np.ndarray, edges: np.ndarray, brot: int) -> OffsetModel:
    from dfthcsos.scoreinfra.dfthc_math import reconstruct_u_vectors, reconstruct_x, split_u_vectors

    vectors = reconstruct_u_vectors(words / (1 << (brot - 1)), edges)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    one_body, two_body = split_u_vectors(vectors / norms, inputs.w)
    del one_body
    x = reconstruct_x(two_body, inputs.w)
    x = np.tril(x) + np.swapaxes(np.tril(x, k=-1), -1, -2)
    tei = np.asarray(inputs.ref["tei"], dtype=np.float64)
    oei = np.asarray(inputs.ref["oei"], dtype=np.float64)
    h1_thc = oei - 0.5 * np.einsum("prrq->pq", tei)
    h0 = (
        h1_thc
        - 0.5 * inputs.ref["electrons"] * inputs.S
        + inputs.beta_1 * np.eye(inputs.num_orbitals)
    )
    return OffsetModel(
        h0=h0,
        x=x,
        w=inputs.w,
        beta_1=inputs.beta_1,
        ecore=float(inputs.ref["ecore"]),
        electrons=int(inputs.ref["electrons"]),
        energy_upper=float(inputs.energy_upper),
        start=inputs.W,
    )


def _serializable(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in metrics.items()
    }


def refit_offsets(
    factorization: dict[str, np.ndarray],
    words: np.ndarray,
    edges: np.ndarray,
    brot: int,
    *,
    radii: tuple[float, ...] = (2e-4, 6e-4, 2e-3, 6e-3, 2e-2),
    maxiter: int = 80,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return the best officially accepted H2_I-only refit and its report."""
    from dfthcsos.verify import evaluate_words, load_inputs, passes_accuracy

    original = {name: np.array(value) for name, value in factorization.items()}
    inputs = load_inputs(original)
    baseline_metrics = evaluate_words(inputs, words, edges, brot)
    if not passes_accuracy(baseline_metrics):
        raise ValueError(f"baseline factorization is not accepted: {baseline_metrics}")

    model = _build_model(inputs, words, edges, brot)
    start = np.array(inputs.W, dtype=np.float64)
    best_params = original
    best_metrics = baseline_metrics
    trials: list[dict[str, Any]] = []

    for radius in radii:
        widths = radius * (1.0 + np.abs(start))
        bounds = list(zip((start - widths).ravel(), (start + widths).ravel()))

        def function(flat: np.ndarray) -> tuple[float, np.ndarray]:
            return model.value_gradient(flat)

        result = minimize(
            function,
            start.ravel(),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-9, "maxls": 30},
        )
        offsets = np.asarray(result.x).reshape(start.shape)
        candidate = {name: np.array(value) for name, value in original.items()}
        candidate["H2_I"] = offsets + np.sum(inputs.w, axis=1)
        candidate_inputs = replace(inputs, params=candidate, W=offsets)
        metrics = evaluate_words(candidate_inputs, words, edges, brot)
        accepted = bool(passes_accuracy(metrics))
        improved = accepted and float(metrics["lambda_eff"]) < float(best_metrics["lambda_eff"])
        trials.append({
            "radius": radius,
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "iterations": int(result.nit),
            "evaluations": int(result.nfev),
            "accepted": accepted,
            "improved": improved,
            "max_offset_change": float(np.max(np.abs(offsets - start))),
            "metrics": _serializable(metrics),
        })
        if improved:
            best_params = candidate
            best_metrics = metrics

    report = {
        "method": "fixed decoded words; H2_I/offset refit only; official metric selection",
        "baseline_metrics": _serializable(baseline_metrics),
        "selected_metrics": _serializable(best_metrics),
        "improved": float(best_metrics["lambda_eff"]) < float(baseline_metrics["lambda_eff"]),
        "trials": trials,
    }
    return best_params, report


def gradient_check(seed: int = 20261025) -> float:
    """Finite-difference check of the analytic objective gradient on a small model."""
    rng = np.random.default_rng(seed)
    r, c, n, b = 2, 3, 5, 4
    x = rng.normal(scale=0.05, size=(r, c, n, n))
    x = 0.5 * (x + np.swapaxes(x, -1, -2))
    model = OffsetModel(
        h0=np.diag(np.linspace(-2.0, 2.0, n)) + rng.normal(scale=0.01, size=(n, n)),
        x=x,
        w=rng.normal(scale=0.03, size=(r, b, c)),
        beta_1=0.0,
        ecore=-10.0,
        electrons=4,
        energy_upper=-7.0,
        start=rng.normal(scale=0.05, size=(r, c)),
    )
    model.h0 = 0.5 * (model.h0 + model.h0.T)
    point = model.start.ravel() + rng.normal(scale=1e-3, size=r * c)
    _, analytic = model.value_gradient(
        point, gap_floor=-1e6, floor_penalty=0.0, trust_penalty=0.0
    )
    numeric = np.zeros_like(point)
    step = 1e-6
    for index in range(len(point)):
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        fp = model.value_gradient(plus, gap_floor=-1e6, floor_penalty=0.0, trust_penalty=0.0)[0]
        fm = model.value_gradient(minus, gap_floor=-1e6, floor_penalty=0.0, trust_penalty=0.0)[0]
        numeric[index] = (fp - fm) / (2 * step)
    return float(np.max(np.abs(analytic - numeric)))


if __name__ == "__main__":
    error = gradient_check()
    print(f"maximum gradient error: {error:.3e}")
    if error > 2e-5:
        raise SystemExit(1)

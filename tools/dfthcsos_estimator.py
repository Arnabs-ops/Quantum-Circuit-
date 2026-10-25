#!/usr/bin/env python3
"""Cheap verifier-aligned resource estimator for DFTHC-SOS candidates.

This models the public four-address signed-angle loader. Its lookup count is
exact for a dense/saturated table and reproduces the public R26 build exactly.
Numerical feasibility must still be checked with the official verifier math.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np


def _import_heuristic(verifier_root: pathlib.Path):
    sys.path.insert(0, str(verifier_root))
    from verifiers.dfthcsos.verify import heuristic_costs

    return heuristic_costs


def qroam_schedule(
    rank_r: int,
    bases_b: int,
    ensemble_c: int,
    brot: int,
    verifier_root: pathlib.Path,
) -> dict[str, int]:
    """Expose the discrete QROM/QROAM block choices hidden in the cost total."""
    sys.path.insert(0, str(verifier_root))
    from verifiers.dfthcsos.scoreinfra.costs_prepare import QI, QR, SpinFreeData_V1

    model = SpinFreeData_V1(
        R=rank_r, B=bases_b, C=ensemble_c, N=76, chi=10, beta=brot
    )
    outer_rows = 76 + rank_r * ensemble_c
    outer_word_bits = math.ceil(math.log2(outer_rows)) + 10
    inner_rows = rank_r * ensemble_c * bases_b
    inner_word_bits = math.ceil(math.log2(bases_b)) + 11
    inner_raw_k = QR(inner_rows, inner_word_bits)[0]
    inner_k = model.find_qroam_size_to_minimize_global_temp_ancilla(inner_word_bits)
    while bases_b < 1 << inner_k:
        inner_k -= 1
    inner_inverse_k = QR(rank_r * ensemble_c, bases_b + 1)[0]
    outer_k = QR(outer_rows, outer_word_bits)[0]
    outer_inverse_k = QI(outer_rows)[0]
    return {
        "outer_rows": outer_rows,
        "outer_word_bits": outer_word_bits,
        "outer_forward_log_block": outer_k,
        "outer_forward_block_size": 1 << outer_k,
        "outer_inverse_log_block": outer_inverse_k,
        "outer_inverse_block_size": 1 << outer_inverse_k,
        "inner_rows": inner_rows,
        "inner_word_bits": inner_word_bits,
        "inner_unconstrained_log_block": inner_raw_k,
        "inner_unconstrained_block_size": 1 << inner_raw_k,
        "inner_selected_log_block": inner_k,
        "inner_selected_block_size": 1 << inner_k,
        "inner_inverse_log_block": inner_inverse_k,
        "inner_inverse_block_size": 1 << inner_inverse_k,
    }


def exact_metrics(
    candidate_npz: pathlib.Path,
    brot: int,
    verifier_root: pathlib.Path,
) -> tuple[dict, bool, tuple[int, int, int]]:
    """Evaluate quantized words against the official FCIDUMP-backed metric."""
    sys.path.insert(0, str(verifier_root))
    from verifiers.dfthcsos.verify import evaluate_words, load_inputs, passes_accuracy
    from dfthcsos.scoreinfra.dfthc_math import reconstruct_u_vectors
    from dfthcsos.scoreinfra.reference import load_fixed_shifts

    with np.load(candidate_npz, allow_pickle=False) as data:
        required = {"W", "H2_I", "angle_words", "tree_edges"}
        if not required <= set(data.files):
            raise ValueError(f"candidate NPZ must contain {sorted(required)}")
        words = np.array(data["angle_words"])
        edges = np.array(data["tree_edges"])
        shape = tuple(map(int, np.shape(data["W"])))
        if len(shape) != 3:
            raise ValueError(f"candidate W must have shape (R,B,C), got {shape}")
        if {"U", "H1_I", "Hsym_lower"} <= set(data.files):
            factorization: pathlib.Path | dict = candidate_npz
        else:
            # Public training checkpoints deliberately omit reconstructible U
            # and the verifier-fixed shifts. Recreate them before any circuits
            # are generated, using the exact same functions as the builder.
            vectors = reconstruct_u_vectors(words / (1 << (brot - 1)), edges)
            factorization = {
                "U": vectors[76:].reshape(shape[0], shape[1], 76),
                "W": np.array(data["W"]),
                "H2_I": np.array(data["H2_I"]),
                **load_fixed_shifts(),
            }
    inputs = load_inputs(factorization)
    metrics = evaluate_words(inputs, words, edges, brot)
    return metrics, passes_accuracy(metrics), shape


def four_address_loader(rows: int, brot: int, batch_size: int) -> dict[str, int | bool]:
    """Resource schedule for the public signed four-address loader."""
    if rows < 8 or brot < 3 or batch_size < 1:
        raise ValueError("rows>=8, brot>=3, and batch_size>=1 are required")
    selection_bits = (rows - 1).bit_length()
    if selection_bits < 8:
        raise ValueError("the public split-ANF eraser requires at least 8 selection bits")

    groups = (rows + 3) // 4
    high_bits = selection_bits - 2
    initialization_toffoli = max(0, groups - high_bits - 1)
    bank_qubits = 3 * batch_size * (brot - 1)
    low_bits = 5
    other_bits = high_bits - low_bits
    eraser_toffoli = (
        (1 << low_bits) - low_bits - 1
        + (1 << other_bits) - other_bits - 1
    )
    per_circuit_toffoli = (
        initialization_toffoli
        + bank_qubits
        + 1
        + eraser_toffoli
    )
    base_width = selection_bits + batch_size * brot + bank_qubits
    scratch = 6 if base_width + 6 <= 128 else 5
    width = base_width + scratch
    circuits = math.ceil(75 / batch_size)
    return {
        "selection_bits": selection_bits,
        "address_capacity": 1 << selection_bits,
        "unused_addresses": (1 << selection_bits) - rows,
        "rows_before_address_width_increases": (1 << selection_bits) - rows,
        "fits_key_address_thresholds": {
            str(limit): rows <= limit for limit in (512, 1024, 2048)
        },
        "grouped_rows": groups,
        "bank_qubits": bank_qubits,
        "scratch_qubits": scratch,
        "circuit_qubits": width,
        "within_128_qubit_cap": width <= 128,
        "num_circuits": circuits,
        "initialization_toffoli_per_circuit": initialization_toffoli,
        "eraser_toffoli_per_circuit": eraser_toffoli,
        "toffoli_per_circuit_per_pass": per_circuit_toffoli,
        "toffoli_both_passes": 2 * circuits * per_circuit_toffoli,
        "persistent_qubits": batch_size * brot,
        "temporary_qubits": bank_qubits + scratch,
    }


def estimate(
    rank_r: int,
    bases_b: int,
    ensemble_c: int,
    brot: int,
    batch_size: int,
    lambda_eff: float,
    *,
    verifier_root: pathlib.Path,
    reference_cost: float = 3355599588.3155375,
    candidate_npz: pathlib.Path | None = None,
    incumbent_cost: float | None = None,
) -> dict:
    """Return loader, official heuristic, and combined score estimates."""
    heuristic_costs = _import_heuristic(verifier_root)
    rows = 76 + rank_r * bases_b
    loader = four_address_loader(rows, brot, batch_size)
    heuristic = heuristic_costs(76, rank_r, bases_b, ensemble_c, brot)
    qroam = qroam_schedule(
        rank_r, bases_b, ensemble_c, brot, verifier_root
    )
    total_t = heuristic["toffoli"] + loader["toffoli_both_passes"]
    total_persistent = heuristic["persistent_qubits"] + loader["persistent_qubits"]
    total_temporary = max(heuristic["temporary_qubits"], loader["temporary_qubits"])
    total_q = total_persistent + total_temporary
    metrics = None
    passes = None
    if candidate_npz is not None:
        metrics, passes, candidate_shape = exact_metrics(candidate_npz, brot, verifier_root)
        requested_shape = (rank_r, bases_b, ensemble_c)
        if candidate_shape != requested_shape:
            raise ValueError(
                f"candidate W shape is {candidate_shape}, but requested (R,B,C) is {requested_shape}"
            )
        lambda_eff = float(metrics["lambda_eff"])
    cap_ok = bool(loader["within_128_qubit_cap"])
    accepted = cap_ok and passes is not False
    cost = float(lambda_eff) * total_t * total_q if accepted else None
    beats_incumbent = (
        None if incumbent_cost is None or cost is None else cost < incumbent_cost
    )
    should_generate = accepted and beats_incumbent is not False
    rejection_reasons = []
    if not cap_ok:
        rejection_reasons.append("four-address loader exceeds the 128-qubit cap")
    if passes is False:
        rejection_reasons.append("quantized factorization fails an official numerical acceptance test")
    if beats_incumbent is False:
        rejection_reasons.append(
            f"estimated cost {cost:.12g} does not beat incumbent {incumbent_cost:.12g}"
        )
    result = {
        "shape": {"R": rank_r, "B": bases_b, "C": ensemble_c},
        "brot": brot,
        "J": batch_size,
        "rows": rows,
        "loader": loader,
        "qroam": qroam,
        "heuristic": heuristic,
        "total": {
            "toffoli": total_t,
            "persistent_qubits": total_persistent,
            "temporary_qubits": total_temporary,
            "qubits": total_q,
            "lambda_eff": float(lambda_eff),
            "cost": cost,
            "reference_over_cost": reference_cost / cost if cost is not None else None,
            "accepted_by_cheap_checks": accepted,
            "incumbent_cost": incumbent_cost,
            "beats_incumbent": beats_incumbent,
            "should_generate_circuits": should_generate,
            "rejection_reasons": rejection_reasons,
        },
    }
    if metrics is not None:
        result["metrics"] = metrics
        result["passes_numerical_acceptance"] = passes
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--C", type=int, required=True)
    parser.add_argument("--brot", type=int, default=10)
    parser.add_argument("--J", type=int, default=3)
    parser.add_argument("--lambda-eff", type=float)
    parser.add_argument("--candidate-npz", type=pathlib.Path)
    parser.add_argument("--verifier-root", type=pathlib.Path, required=True)
    parser.add_argument("--incumbent-cost", type=float)
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="exit with status 2 when cheap checks say not to generate circuits",
    )
    args = parser.parse_args()
    if args.lambda_eff is None and args.candidate_npz is None:
        parser.error("provide --lambda-eff or --candidate-npz")
    result = estimate(
        args.R,
        args.B,
        args.C,
        args.brot,
        args.J,
        args.lambda_eff if args.lambda_eff is not None else 1.0,
        verifier_root=args.verifier_root,
        candidate_npz=args.candidate_npz,
        incumbent_cost=args.incumbent_cost,
    )
    print(json.dumps(result, indent=2))
    if args.require_improvement and not result["total"]["should_generate_circuits"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

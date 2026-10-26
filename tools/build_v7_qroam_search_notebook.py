#!/usr/bin/env python3
"""Build the guarded v7 DFTHC dimension-search Kaggle notebook."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "high_scoring_quantum_circuit_compilation_v5_portfolio.ipynb"
OUTPUT = ROOT / "high_scoring_quantum_circuit_compilation_v7_qroam_search.ipynb"


def lines(text: str) -> list[str]:
    if text and not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(source),
    }


INTRO = r"""# Portfolio v7: quantized DFTHC dimension search

The v6 offset-only experiment scored **10.88057**, essentially unchanged from
10.88048. That result closes the offset-only route. This notebook changes the
factorization dimensions and searches the resource discontinuities that the
score actually rewards.

It starts from the fully verified Q82 inverse-square-root, Q898 DQI and R26
chemistry portfolio. The chemistry search contribution-ranks groups, basis
vectors and ensemble coordinates; retrains with the official starter-kit
trainer; quantizes to ten-bit Givens words; freezes those decoded vectors for a
weight refit; and applies the authoritative numerical predicates. Only an
accepted candidate whose exact circuit resources beat the incumbent chemistry
cost can replace it. The final selected portfolio is then verified at full shot
counts and packaged. If every experiment fails, the notebook emits the same
verified score-10 portfolio.

The default `TRAIN_PRESET="full"` runs the near score-11 frontier and the
larger address/QROAM experiments. `"score11"` stops after the R26/B46 and R20
branches for a shorter run.

| Candidate | Why it is included |
|---|---|
| R26/B49/C30 | About 11.02 projected at the incumbent lambda with the public loader |
| R26/B46/C30 or R20/B60/C30 | About 11.15–11.20 projected |
| R19/B49/C24 | Crosses the 1,024-row address boundary; about 11.83 projected |
| R12/B36/C20 | Crosses the inner-QROAM block-16 cliff; about 16.0 projected |

Projections are resource estimates. Only the exact quantized verifier result
shown at the end is a valid competition result.
"""


SETUP = r'''import hashlib
import json
import math
import os
import pathlib
import platform
import shutil
import sys
import threading
import time
import zipfile

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    XLA_PYTHON_CLIENT_PREALLOCATE="false",
)

import numpy as np
import pandas as pd
from IPython.display import display

INPUT = pathlib.Path(os.environ.get("QCC_INPUT", "/kaggle/input"))
WORK = pathlib.Path(os.environ.get("QCC_WORK", "/kaggle/working"))
SUB = pathlib.Path(os.environ.get("QCC_SUB", "/tmp/qcc_submission_v7"))
SLUG = "google-verifiable-quantum-circuit-compilation"
REFERENCE = {
    "invsqrt": 2076774.233715282,
    "dqi": 24439474177519.598,
    "dfthcsos": 3355599588.3155365,
}
PLAN = {
    "invsqrt": {
        "dataset_version_id": 21216223,
        "slug": "qcc-best-invsqrt-circuit",
        "rel": "invsqrt.kmb",
        "sha256": "0f84f5bab3a02e6b357d4656553607174345a33d252cef643f9136db13212141",
    },
    "dqi": {
        "dataset_version_id": 21211427,
        "slug": "qcc-best-dqi",
        "rel": "dqi.kmb",
        "sha256": "c5da9cd4e2d73dd41417e539f2659009d646b3ee30ddb5ca494261378a502856",
    },
    "dfthcsos": {
        "dataset_version_id": 21216621,
        "slug": "qcc-best-circuits",
        "rel": ".",
        "sha256": "47187a5d3664c76e48f1266c39a24b55a7bcc845743d6a673c26c8e69b5cbcee",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(path):
    path = pathlib.Path(path)
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode())
            digest.update(child.read_bytes())
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def competition_candidates(root):
    preferred = [root, root / "competitions" / SLUG, root / SLUG]
    discovered = [path.parent for path in root.rglob("kickmix") if path.is_dir()]
    unique = []
    for path in preferred + sorted(discovered):
        path = path.resolve()
        if path not in unique and (path / "kickmix").is_dir() and (path / "verifiers").is_dir():
            unique.append(path)
    return unique


def unique_dataset_root(slug):
    hits = sorted(path.resolve() for path in INPUT.rglob(slug)
                  if path.is_dir() and path.name == slug)
    unique = list(dict.fromkeys(hits))
    if len(unique) != 1:
        raise RuntimeError(f"Expected exactly one {slug!r} dataset directory; found {unique}")
    return unique[0]


if sys.version_info[:2] != (3, 13):
    raise RuntimeError(f"kickmix.so requires CPython 3.13; found {platform.python_version()}")
competition_dirs = competition_candidates(INPUT)
if len(competition_dirs) != 1:
    raise RuntimeError(f"Expected one competition input; found {competition_dirs}")
COMP = competition_dirs[0]

SOURCES = {}
for task, entry in PLAN.items():
    root = unique_dataset_root(entry["slug"])
    source = (root / entry["rel"]).resolve()
    observed = content_sha256(source)
    if observed != entry["sha256"]:
        raise AssertionError(f"{task}: expected {entry['sha256']}, got {observed}")
    SOURCES[task] = source
    print(f"{task:9s} dataset v{entry['dataset_version_id']} hash OK")

starter_hits = sorted(path.resolve() for path in INPUT.rglob("starter_kit")
                      if (path / "dfthc" / "train_df_thc_sga.py").is_file())
starter_hits = list(dict.fromkeys(starter_hits))
if len(starter_hits) != 1:
    raise RuntimeError(f"Expected starter-kit dataset version 21107076; found {starter_hits}")
STARTER = starter_hits[0]

WORK.mkdir(parents=True, exist_ok=True)
shutil.rmtree(SUB, ignore_errors=True)
(SUB / "dfthcsos").mkdir(parents=True)
shutil.copy2(SOURCES["invsqrt"], SUB / "invsqrt.kmb")
shutil.copy2(SOURCES["dqi"], SUB / "dqi.kmb")
chem_source = SOURCES["dfthcsos"]
for source in sorted(path for path in chem_source.iterdir()
                     if path.name == "submission.npz" or
                     (path.name.startswith("angle_k_") and path.suffix == ".kmb")):
    shutil.copy2(source, SUB / "dfthcsos" / source.name)

sys.path[:0] = [str(COMP / "kickmix"), str(COMP / "verifiers"), str(STARTER)]
import kickmix as km

available_cpus = os.cpu_count() or 4
WORKERS = max(1, min(int(os.environ.get("QCC_WORKERS", available_cpus)), available_cpus, 4))
print(f"Competition data: {COMP}")
print(f"Starter kit     : {STARTER}")
print(f"Submission tree : {SUB}")
print(f"Workers / CPUs  : {WORKERS} / {available_cpus}")
'''


def sources_cell(dimension_source: str, loader_source: str, offset_source: str) -> str:
    return f'''# Install the reviewed source modules inside this run.
SOURCE_ROOT = pathlib.Path("/tmp/qcc_v7_source")
shutil.rmtree(SOURCE_ROOT, ignore_errors=True)
(SOURCE_ROOT / "dfthcsos" / "group_loader").mkdir(parents=True)
(SOURCE_ROOT / "data").mkdir(parents=True)
os.symlink(COMP, SOURCE_ROOT / "data" / "comp", target_is_directory=True)
(SOURCE_ROOT / "dfthcsos" / "group_loader" / "four_address.py").write_text({loader_source!r})
(SOURCE_ROOT / "dfthcsos_dimension_train.py").write_text({dimension_source!r})
(SOURCE_ROOT / "dfthcsos_offset_refit.py").write_text({offset_source!r})
sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "dfthcsos" / "group_loader")]

import dfthcsos_dimension_train as DT
import dfthcsos_offset_refit as OR
import four_address as FOUR
DT.self_test()
assert OR.gradient_check() < 2e-5

from dfthc.train_df_thc_sga import Initialize, Train
from dfthc import df_thc_sga
from starter.task2_angles.encode import encode_inputs
from starter.task2_angles.trees import log_depth_tree_edges
from verifier.score import load_inputs as kit_load_inputs
import jax
import jax.numpy as jnp
from dfthcsos import verify as VC
from dfthcsos.example_submission.angle_k_reference_circuit import batch_deltas
from dfthcsos.scoreinfra.dfthc_math import reconstruct_u_vectors

print("JAX backend:", jax.default_backend(), jax.devices())
print("Loading the official FeMoCo reference", flush=True)
REFERENCE_DATA = Initialize(str(COMP / "verifiers" / "dfthcsos" / "FeMoCo76o113e.fcidump"),
                            -22140.27335)
TREE_EDGES = np.asarray(log_depth_tree_edges(76), dtype=np.int64)


def load_circuit(path):
    return km.Circuit.from_file(str(path), format="kmb")


BASE_CIRCUITS = [load_circuit(path) for path in sorted(
    (SUB / "dfthcsos").glob("angle_k_*.kmb"),
    key=lambda path: int(path.stem.rsplit("_", 1)[1]),
)]
with np.load(SUB / "dfthcsos" / "submission.npz", allow_pickle=False) as archive:
    BASE_ARCHIVE = {{name: np.array(archive[name]) for name in archive.files}}
base_edges = np.asarray(BASE_ARCHIVE.pop("tree_edges"))
if not np.array_equal(base_edges, TREE_EDGES):
    raise AssertionError("verified R26 checkpoint uses an unexpected Givens tree")
base_inputs = VC.load_inputs(BASE_ARCHIVE)
base_quick = VC.evaluate_circuits(base_inputs, BASE_CIRCUITS, base_edges,
                                  shots_per_address=1, seed=20261026)
BASE_WORDS = np.asarray(base_quick["angle_words"])
BASE_BROT = int(base_quick["brot"])

# Retain the small v6 offset improvement before using the checkpoint as a seed.
BASE_PARAMETERS, BASE_OFFSET_REPORT = OR.refit_offsets(
    BASE_ARCHIVE, BASE_WORDS, base_edges, BASE_BROT,
    radii=(2e-4, 6e-4, 2e-3, 6e-3, 2e-2), maxiter=80,
)
BASE_PARAMETERS = DT.numpy_parameters(BASE_PARAMETERS)
base_metrics = BASE_OFFSET_REPORT["selected_metrics"]
INCUMBENT_CHEM_COST = VC.score_from_metrics(base_metrics, base_quick["resources"])
if BASE_OFFSET_REPORT["improved"]:
    baseline_payload = {{name: np.asarray(value) for name, value in BASE_PARAMETERS.items()}}
    baseline_payload["tree_edges"] = base_edges
    baseline_payload["angle_words"] = BASE_WORDS
    np.savez(SUB / "dfthcsos" / "submission.npz", **baseline_payload)
print(f"Chemistry incumbent after guarded offset refit: {{INCUMBENT_CHEM_COST:.9g}}")
'''


TRAIN = r'''# Set to "score11" to omit the two long high-upside branches.
TRAIN_PRESET = "full"

SEARCH_PLAN = [
    dict(name="b56", shape=(26, 56, 30), parent="baseline", epochs=(8000, 12000)),
    dict(name="b52", shape=(26, 52, 30), parent="b56", epochs=(10000, 14000)),
    dict(name="b49", shape=(26, 49, 30), parent="b52", epochs=(12000, 18000)),
    dict(name="b46", shape=(26, 46, 30), parent="b49", epochs=(14000, 20000)),
    dict(name="r24", shape=(24, 60, 30), parent="baseline", epochs=(10000, 14000)),
    dict(name="r22", shape=(22, 60, 30), parent="r24", epochs=(12000, 18000)),
    dict(name="r20", shape=(20, 60, 30), parent="r22", epochs=(14000, 22000)),
]
if TRAIN_PRESET == "full":
    SEARCH_PLAN += [
        dict(name="a1024", shape=(19, 49, 24), parent="r20", epochs=(22000, 30000)),
        dict(name="qroam16", shape=(12, 36, 20), parent="a1024", epochs=(32000, 45000)),
    ]
elif TRAIN_PRESET != "score11":
    raise ValueError(f"unknown TRAIN_PRESET={TRAIN_PRESET!r}")


def native(params):
    return {name: np.asarray(value) for name, value in DT.numpy_parameters(params).items()}


def trainer_arrays(params):
    return {name: jnp.asarray(value, dtype=df_thc_sga.dtype)
            for name, value in native(params).items()}


def train_phase(params, *, epochs, rho, learning_rate, learning_rate_final,
                freeze_u=False, label="phase"):
    p = trainer_arrays(params)
    frozen_names = ["H1_I", "Hsym_lower"] + (["U"] if freeze_u else [])
    frozen = {name: p[name] for name in frozen_names}
    print(f"{label}: epochs={epochs} rho={rho:g} freeze_u={freeze_u}", flush=True)
    solution = Train(
        generating_set="Qaggle",
        method="DF-THC-SOS",
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        learning_rate_final=float(learning_rate_final),
        rho=float(rho),
        params=p,
        ref=REFERENCE_DATA,
        print_interval=max(1000, int(epochs) // 5),
        epsilon_dfthc=0.008,
        frozen_params=frozen,
        lambda_floor=49.0,
        sos_gap_floor=3.914898,
    )
    result = native(solution["params_best"])
    if freeze_u:
        result["U"] = np.asarray(params["U"]).copy()
    return result, {name: float(value) for name, value in solution["values"].items()
                    if np.ndim(value) == 0}


def canonicalize_words(words, edges):
    words = np.asarray(words, dtype=np.int64)
    vectors = reconstruct_u_vectors(words / 512, edges)
    remaining = vectors.copy()
    angles = np.zeros_like(words, dtype=np.float64)
    for j in reversed(range(len(edges))):
        parent, child = edges[j]
        norm = np.hypot(remaining[:, parent], remaining[:, child])
        sign = np.where(np.abs(remaining[:, parent]) < 1e-12,
                        np.where(remaining[:, child] > 0, -1.0, 1.0),
                        np.where(remaining[:, parent] < 0, -1.0, 1.0))
        angles[:, j] = np.where(
            norm < 1e-12, 0.0,
            np.arctan2(remaining[:, child] * sign, remaining[:, parent] * sign),
        )
        remaining[:, parent] = np.where(norm < 1e-12, 0.0, norm * sign)
        remaining[:, child] = 0.0
    signed = np.rint(angles * (512 / np.pi)).astype(np.int64) % 1024
    if np.any(((signed >> 9) ^ (signed >> 8)) & 1):
        raise ArithmeticError("signed ten-bit gauge failed")
    decoded = reconstruct_u_vectors(signed / 512, edges)
    row_sign = np.where(np.sum(vectors * decoded, axis=1) < 0, -1.0, 1.0)
    if float(np.max(np.abs(decoded * row_sign[:, None] - vectors))) > 1e-10:
        raise ArithmeticError("signed gauge changed a decoded vector")
    return signed, decoded


def encode_payload(params):
    params = native(params)
    kit_inputs = kit_load_inputs("FeMoCo76o113e", params)
    words = np.asarray(encode_inputs(kit_inputs, TREE_EDGES, brot=10, rounding="nearest"),
                       dtype=np.int64)
    words, decoded = canonicalize_words(words, TREE_EDGES)
    shape = params["W"].shape
    params["U"] = decoded[76:].reshape(shape[0], shape[1], 76)
    return params, words


def exact_metrics(params, words):
    inputs = VC.load_inputs(params)
    metrics = VC.evaluate_words(inputs, words, TREE_EDGES, 10)
    return inputs, metrics


def build_candidate(name, params, words):
    out = pathlib.Path("/tmp/qcc_v7_candidates") / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    circuits, reports = [], []
    for index, delta in enumerate(batch_deltas(words, 3)):
        delta = np.asarray(delta, dtype=np.int64)
        if len(delta) % 4:
            delta = np.pad(delta, ((0, 4 - len(delta) % 4), (0, 0)))
        circuit, report = FOUR.build(delta, brot=10, return_report=True)
        if circuit.num_qubits > 128:
            raise ValueError(f"{name}: ANGLE_{index} uses {circuit.num_qubits} qubits")
        circuit.to_file(str(out / f"angle_k_{index}.kmb"), format="kmb")
        circuits.append(circuit)
        reports.append(report)
    payload = {key: np.asarray(value) for key, value in native(params).items()}
    payload["tree_edges"] = TREE_EDGES
    payload["angle_words"] = words
    np.savez(out / "submission.npz", **payload)
    inputs, metrics = exact_metrics(payload, words)
    angle_toffoli = 2 * sum(c.operation_statistics().num_toffoli_like for c in circuits)
    resources = VC.resource_costs(inputs, circuits, 10, angle_toffoli)
    score = VC.score_from_metrics(metrics, resources)
    return out, metrics, resources, float(score), reports


def serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


parents = {"baseline": native(BASE_PARAMETERS)}
TRAINING_REPORT = {
    "preset": TRAIN_PRESET,
    "incumbent_chemistry_cost": INCUMBENT_CHEM_COST,
    "base_offset_refit": BASE_OFFSET_REPORT,
    "trials": [],
}
accepted_candidates = []

for spec in SEARCH_PLAN:
    name, shape = spec["name"], tuple(spec["shape"])
    print(f"\n=== {name}: target R/B/C={shape} from {spec['parent']} ===", flush=True)
    trial = {"name": name, "shape": list(shape), "parent": spec["parent"]}
    if spec["parent"] not in parents:
        trial["skipped"] = f"parent {spec['parent']} did not finish"
        TRAINING_REPORT["trials"].append(trial)
        print(f"{name}: skipped because parent {spec['parent']} is unavailable", flush=True)
        continue
    try:
        parent = parents[spec["parent"]]
        pruned, pruning = DT.prune_dimensions(parent, *shape)
        recovery_epochs, lambda_epochs = spec["epochs"]
        trained, recovery_values = train_phase(
            pruned, epochs=recovery_epochs, rho=1e-6,
            learning_rate=3e-4, learning_rate_final=5e-5,
            label=f"{name}/accuracy",
        )
        trained, lambda_values = train_phase(
            trained, epochs=lambda_epochs, rho=1e-5,
            learning_rate=1e-4, learning_rate_final=1e-5,
            label=f"{name}/lambda",
        )
        parents[name] = trained
        np.savez(WORK / f"v7_continuous_{name}.npz", **trained)

        payload, words = encode_payload(trained)
        _, before_metrics = exact_metrics(payload, words)
        quantized_refit = None
        # Refit W/H2_I with the quantized vectors fixed. If a starter-kit
        # revision cannot freeze U, retain the already evaluated candidate.
        try:
            payload, quantized_refit = train_phase(
                payload, epochs=max(6000, lambda_epochs // 2), rho=1e-5,
                learning_rate=5e-5, learning_rate_final=5e-6,
                freeze_u=True, label=f"{name}/fixed-word-refit",
            )
        except Exception as exc:
            quantized_refit = {"skipped": type(exc).__name__, "message": str(exc)}
            print(f"{name}: fixed-word trainer refit skipped: {exc}", flush=True)

        _, refit_metrics = exact_metrics(payload, words)
        if VC.passes_accuracy(refit_metrics):
            payload, offset_report = OR.refit_offsets(
                payload, words, TREE_EDGES, 10,
                radii=(2e-4, 6e-4, 2e-3), maxiter=50,
            )
        else:
            offset_report = {"improved": False, "skipped": "pre-offset metrics rejected"}
        _, final_metrics = exact_metrics(payload, words)

        trial.update(
            pruning=pruning,
            recovery_values=recovery_values,
            lambda_values=lambda_values,
            metrics_before_fixed_refit=serializable(before_metrics),
            fixed_word_refit=quantized_refit,
            offset_refit=offset_report,
            final_metrics=serializable(final_metrics),
            numerical_pass=bool(VC.passes_accuracy(final_metrics)),
        )
        print(
            f"{name}: e1={1e3*final_metrics['error_1']:.3f}mHa "
            f"e2={1e3*final_metrics['error_2']:.3f}mHa "
            f"lambda={final_metrics['lambda_eff']:.5f} "
            f"accepted={trial['numerical_pass']}", flush=True,
        )
        if trial["numerical_pass"]:
            out, metrics, resources, cost, build_reports = build_candidate(name, payload, words)
            trial.update(
                resources=serializable(resources),
                exact_projected_cost=cost,
                beats_incumbent=bool(cost < INCUMBENT_CHEM_COST),
                build_reports=serializable(build_reports),
            )
            print(
                f"{name}: Q={resources['qubits']} T={resources['toffoli']:.0f} "
                f"cost={cost:.8g} beats incumbent={cost < INCUMBENT_CHEM_COST}",
                flush=True,
            )
            if cost < INCUMBENT_CHEM_COST:
                accepted_candidates.append((cost, name, out))
    except Exception as exc:
        trial.update(error_type=type(exc).__name__, error=str(exc))
        print(f"{name}: experiment failed safely: {type(exc).__name__}: {exc}", flush=True)
    TRAINING_REPORT["trials"].append(trial)
    (WORK / "v7_training_report.partial.json").write_text(
        json.dumps(serializable(TRAINING_REPORT), indent=2, default=str) + "\n"
    )

SELECTED_CHEMISTRY = "incumbent"
SELECTED_CHEMISTRY_COST = float(INCUMBENT_CHEM_COST)
if accepted_candidates:
    SELECTED_CHEMISTRY_COST, SELECTED_CHEMISTRY, selected_dir = min(accepted_candidates)
    shutil.rmtree(SUB / "dfthcsos")
    shutil.copytree(selected_dir, SUB / "dfthcsos")
    print(f"Selected {SELECTED_CHEMISTRY}: {INCUMBENT_CHEM_COST:.8g} -> "
          f"{SELECTED_CHEMISTRY_COST:.8g}", flush=True)
else:
    print("No accepted structural candidate beat the incumbent; retaining it", flush=True)

TRAINING_REPORT["selected"] = SELECTED_CHEMISTRY
TRAINING_REPORT["selected_chemistry_cost"] = SELECTED_CHEMISTRY_COST
(WORK / "v7_training_report.json").write_text(
    json.dumps(serializable(TRAINING_REPORT), indent=2, default=str) + "\n"
)

# Release trainer state before the full simulators allocate their worker pools.
parents.clear()
accepted_candidates.clear()
del REFERENCE_DATA, BASE_CIRCUITS, BASE_ARCHIVE, base_inputs, base_quick
try:
    jax.clear_caches()
except Exception:
    pass
import gc
gc.collect()
'''


VERIFY = r'''def with_heartbeat(label, function, every=60):
    finished = threading.Event()
    started = time.monotonic()
    def heartbeat():
        while not finished.wait(every):
            print(f"    {label} still running ({(time.monotonic()-started)/60:.1f} min)", flush=True)
    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        return function()
    finally:
        finished.set()
        monitor.join()


started = time.monotonic()
print("[1/3] Full inverse-square-root verification", flush=True)
from invsqrt import verify_submission_invsqrt as VI
invsqrt_result = with_heartbeat(
    "inverse-square-root verifier",
    lambda: VI.verify(load_circuit(SUB / "invsqrt.kmb"),
                      VI.ProblemSpecInvSqrt.for_n_p(10), max_workers=WORKERS),
)
assert invsqrt_result["status"] == "PASS" and invsqrt_result["verified_shots"] == 200540928
assert invsqrt_result["num_qubits"] == 82

print("[2/3] Full DQI verification", flush=True)
from dqi import verify_submission_dqi as VD
dqi_circuit = load_circuit(SUB / "dqi.kmb")
dqi_trials = []
for salt in ("", "independent-v7-a", "independent-v7-b", "independent-v7-c"):
    result = with_heartbeat(
        f"DQI verifier salt={salt or '<empty>'}",
        lambda salt=salt: VD.verify(dqi_circuit, VD.ProblemSpecDQI.benchmark_instance(),
                                    salt=salt, max_workers=WORKERS),
    )
    assert result["status"] == "PASS" and result["verified_shots"] == 102400
    assert result["num_qubits"] == 898
    dqi_trials.append({"salt": salt, **result})
    print(f"    salt={salt or '<empty>'}: value={result['value_errors']} "
          f"phase={result['phase_errors']} R={result['repetition_factor']:.6f} "
          f"cost={result['score']:.7g}", flush=True)
dqi_result = dqi_trials[0]
dqi_worst_result = max(dqi_trials, key=lambda result: result["score"])

print("[3/3] Full DFTHC-SOS verification", flush=True)
chem_paths = sorted((SUB / "dfthcsos").glob("angle_k_*.kmb"),
                    key=lambda path: int(path.stem.rsplit("_", 1)[1]))
circuits = [load_circuit(path) for path in chem_paths]
with np.load(SUB / "dfthcsos" / "submission.npz", allow_pickle=False) as archive:
    factorization = {name: np.array(archive[name]) for name in archive.files}
tree_edges = factorization.pop("tree_edges")
shape = tuple(map(int, factorization["W"].shape))
dfthcsos_result = with_heartbeat(
    "chemistry verifier", lambda: VC.verify(circuits, factorization, tree_edges),
)
assert dfthcsos_result["status"] == "PASS"
assert dfthcsos_result["verified_shots"] == (76 + shape[0] * shape[1]) * 64

RESULTS = {"invsqrt": invsqrt_result, "dqi": dqi_result, "dfthcsos": dfthcsos_result}
SCORE = math.prod(REFERENCE[name] / RESULTS[name]["score"] for name in REFERENCE) ** (1 / 3)
WORST_SALT_SCORE = (
    (REFERENCE["invsqrt"] / invsqrt_result["score"])
    * (REFERENCE["dqi"] / dqi_worst_result["score"])
    * (REFERENCE["dfthcsos"] / dfthcsos_result["score"])
) ** (1 / 3)
if min(SCORE, WORST_SALT_SCORE) < 10.8:
    raise AssertionError(f"v7 fell below safety floor: {SCORE:.6f}, {WORST_SALT_SCORE:.6f}")

rows = []
for name, label in (("invsqrt", "Inverse square root"), ("dqi", "DQI decoder"),
                    ("dfthcsos", f"FeMoCo {shape}")):
    result = RESULTS[name]
    rows.append({
        "Task": label,
        "Qubits": result["num_qubits"],
        "Toffoli": f"{result['toffoli_count']:,.3f}",
        "Cost": f"{result['score']:.7g}",
        "Reference / cost": f"{REFERENCE[name] / result['score']:.3f}",
    })
display(pd.DataFrame(rows).style.hide(axis="index"))
print(f"Selected chemistry: {SELECTED_CHEMISTRY}")
print(f"Full verification completed in {(time.monotonic()-started)/60:.1f} min")
print(f"Verified local score: {SCORE:.5f}")
print(f"Worst reviewed-salt score: {WORST_SALT_SCORE:.5f}")
'''


PACKAGE = r'''MAX_ID = 2**30 // 1536
problems = []
all_files = sorted(path.relative_to(SUB).as_posix() for path in SUB.rglob("*") if path.is_file())
angles = sorted(
    (name for name in all_files if name.startswith("dfthcsos/angle_k_") and name.endswith(".kmb")),
    key=lambda name: int(pathlib.Path(name).stem.rsplit("_", 1)[1]),
)
expected = {"invsqrt.kmb", "dqi.kmb", "dfthcsos/submission.npz"} | set(angles)
if set(all_files) != expected:
    problems.append(f"manifest differs: {all_files}")
if [int(pathlib.Path(name).stem.rsplit("_", 1)[1]) for name in angles] != list(range(len(angles))):
    problems.append("chemistry circuit indices are not contiguous")
for name in sorted(item for item in all_files if item.endswith(".kmb")):
    circuit = load_circuit(SUB / name)
    top_id = max(circuit.num_qubits, circuit.num_bits) - 1
    unnamed = [register for register in circuit.register_data if not isinstance(register, str)]
    if top_id >= MAX_ID:
        problems.append(f"{name}: largest ID {top_id} >= {MAX_ID}")
    if unnamed:
        problems.append(f"{name}: unnamed registers {unnamed}")
with np.load(SUB / "dfthcsos" / "submission.npz", allow_pickle=False) as archive:
    for name in archive.files:
        value = archive[name]
        expected_kind = np.integer if name in {"tree_edges", "angle_words"} else np.floating
        little = value.dtype.byteorder in {"<", "|"} or (
            value.dtype.byteorder == "=" and sys.byteorder == "little")
        if not np.issubdtype(value.dtype, expected_kind) or not little:
            problems.append(f"submission.npz {name}: invalid dtype {value.dtype.str}")
if problems:
    raise AssertionError("\n".join(problems))

ZIP_PATH = WORK / "submission.zip"
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(SUB.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(SUB).as_posix())
with zipfile.ZipFile(ZIP_PATH) as archive:
    infos = archive.infolist()
    if {info.filename for info in infos} != expected or archive.testzip() is not None:
        raise AssertionError("ZIP validation failed")
    member_hashes = {info.filename: hashlib.sha256(archive.read(info.filename)).hexdigest()
                     for info in infos}

summary = {
    "score": SCORE,
    "worst_reviewed_salt_score": WORST_SALT_SCORE,
    "selected_chemistry": SELECTED_CHEMISTRY,
    "results": RESULTS,
    "dqi_salt_trials": dqi_trials,
    "training_report": TRAINING_REPORT,
    "source_plan": PLAN,
    "member_sha256": member_hashes,
    "archive_sha256": sha256_file(ZIP_PATH),
    "archive_bytes": ZIP_PATH.stat().st_size,
}
(WORK / "summary_v7.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
(WORK / "submission-members-v7.sha256").write_text(
    "".join(f"{member_hashes[name]}  {name}\n" for name in sorted(member_hashes))
)
print(f"Submission ready: {ZIP_PATH}")
print(f"Selected        : {SELECTED_CHEMISTRY}")
print(f"Archive SHA-256 : {summary['archive_sha256']}")
print(f"Verified score  : {SCORE:.5f}")
ZIP_PATH
'''


NOTES = r"""## How to read the result

The decisive lines are `Selected chemistry`, `Verified local score`, and the
final full-verifier table. A rejected training branch is recorded in
`v7_training_report.json` and does not affect the submission. Continuous
trainer loss is never used as acceptance evidence.

The output directory also contains every continuous checkpoint, the partial
report written after each branch, `summary_v7.json`, the member hash manifest,
and `submission.zip`. This makes a long run resumable and preserves enough
evidence to decide which shape should receive a longer follow-up training run.
"""


def main() -> None:
    base = json.loads(BASE.read_text())
    dimension_source = (ROOT / "tools" / "dfthcsos_dimension_train.py").read_text()
    loader_source = (ROOT / "tools" / "dfthcsos_four_address.py").read_text()
    offset_source = (ROOT / "tools" / "dfthcsos_offset_refit.py").read_text()
    base["cells"] = [
        markdown(INTRO),
        code(SETUP),
        code(sources_cell(dimension_source, loader_source, offset_source)),
        code(TRAIN),
        code(VERIFY),
        code(PACKAGE),
        markdown(NOTES),
    ]
    kaggle = base.setdefault("metadata", {}).setdefault("kaggle", {})
    sources = kaggle.setdefault("dataSources", [])
    if not any(item.get("sourceType") == "datasetVersion" and item.get("sourceId") == 21107076
               for item in sources):
        sources.append({"sourceType": "datasetVersion", "sourceId": 21107076})
    kaggle.update(
        accelerator="nvidiaTeslaT4",
        isGpuEnabled=True,
        isInternetEnabled=False,
        dockerImageVersionId=31498,
    )
    OUTPUT.write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()

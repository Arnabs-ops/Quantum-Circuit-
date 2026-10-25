#!/usr/bin/env python3
"""Build the verifier-first score-10 portfolio notebook."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "high_scoring_quantum_circuit_compilation_v5_portfolio.ipynb"


def lines(text: str) -> list[str]:
    if text and not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": lines(source)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


INTRO = r"""# Verifier-first score-10 portfolio

This notebook reviews and packages the three exact public artifacts referenced
by **QCC | Verified Portfolio + DF-THC Retraining Study**. The supplied study
reports a local score of **10.864**:

| task | selected artifact | reported cost | reported factor |
|---|---|---:|---:|
| inverse square root | EXP-225a, fused phase, Q82 | 138,095.135 | 15.039× |
| DQI | EXP-119a, paired J-fraction extraction, Q898 | 1.045962e12 | 23.365× |
| DFTHC-SOS | EXP-231c, R26 fouraddr2 | 9.196081e8 | 3.649× |

The original study only checks hashes and writes a ZIP. This review version
also runs all three authoritative verifiers at their full shot counts, checks
the metric's register, ID, manifest, dtype and CRC constraints, and packages
only the bytes that passed in this run. DQI results can move with the secret
leaderboard salt.

The three artifact datasets are pinned by Kaggle dataset-version ID and by the
content hashes published in the supplied study. Use Kaggle's Python 3.13 CPU
image, keep internet off, and select **Restart & Run All**.
"""


PREFLIGHT = r'''import hashlib
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
)

import numpy as np
import pandas as pd
from IPython.display import display

INPUT = pathlib.Path(os.environ.get("QCC_INPUT", "/kaggle/input"))
WORK = pathlib.Path(os.environ.get("QCC_WORK", "/kaggle/working"))
SUB = pathlib.Path(os.environ.get("QCC_SUB", "/tmp/qcc_submission_v5"))
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
        "reported_cost": 138095.13529214897,
    },
    "dqi": {
        "dataset_version_id": 21211427,
        "slug": "qcc-best-dqi",
        "rel": "dqi.kmb",
        "sha256": "c5da9cd4e2d73dd41417e539f2659009d646b3ee30ddb5ca494261378a502856",
        "reported_cost": 1045962255881.3461,
    },
    "dfthcsos": {
        "dataset_version_id": 21216621,
        "slug": "qcc-best-circuits",
        "rel": ".",
        "sha256": "47187a5d3664c76e48f1266c39a24b55a7bcc845743d6a673c26c8e69b5cbcee",
        "reported_cost": 919608140.5572633,
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(path):
    """Match the portfolio study's published file/folder digest exactly."""
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
    hits = sorted(
        path.resolve() for path in INPUT.rglob(slug)
        if path.is_dir() and path.name == slug
    )
    unique = list(dict.fromkeys(hits))
    if len(unique) != 1:
        raise RuntimeError(f"Expected exactly one {slug!r} dataset directory; found {unique}")
    return unique[0]


if sys.version_info[:2] != (3, 13):
    raise RuntimeError(f"kickmix.so requires CPython 3.13; found {platform.python_version()}")
if not INPUT.exists():
    raise FileNotFoundError(INPUT)
competition_dirs = competition_candidates(INPUT)
if len(competition_dirs) != 1:
    raise RuntimeError(f"Expected one competition input; found {competition_dirs}")
COMP = competition_dirs[0]

SOURCES = {}
for task, entry in PLAN.items():
    root = unique_dataset_root(entry["slug"])
    source = (root / entry["rel"]).resolve()
    if not source.exists():
        raise FileNotFoundError(f"{task}: expected {source}")
    observed = content_sha256(source)
    if observed != entry["sha256"]:
        raise AssertionError(
            f"{task}: dataset bytes changed; expected {entry['sha256']}, got {observed}"
        )
    SOURCES[task] = source
    print(f"{task:9s} dataset v{entry['dataset_version_id']} hash OK: {observed}")

WORK.mkdir(parents=True, exist_ok=True)
shutil.rmtree(SUB, ignore_errors=True)
(SUB / "dfthcsos").mkdir(parents=True)
shutil.copy2(SOURCES["invsqrt"], SUB / "invsqrt.kmb")
shutil.copy2(SOURCES["dqi"], SUB / "dqi.kmb")

chem_source = SOURCES["dfthcsos"]
submission_files = sorted(chem_source.glob("submission.npz"))
angle_files = sorted(
    chem_source.glob("angle_k_*.kmb"),
    key=lambda path: int(path.stem.rsplit("_", 1)[1]),
)
if len(submission_files) != 1:
    raise RuntimeError(f"Expected one chemistry submission.npz; found {submission_files}")
indices = [int(path.stem.rsplit("_", 1)[1]) for path in angle_files]
if indices != list(range(len(indices))):
    raise RuntimeError(f"Chemistry angle circuit indices are not contiguous: {indices}")
for source in [*angle_files, submission_files[0]]:
    shutil.copy2(source, SUB / "dfthcsos" / source.name)
if len(angle_files) != 25:
    raise RuntimeError(f"Expected 25 R26 angle circuits, found {len(angle_files)}")

sys.path[:0] = [str(COMP / "kickmix"), str(COMP / "verifiers")]
import kickmix as km

available_cpus = os.cpu_count() or 4
WORKERS = max(1, min(int(os.environ.get("QCC_WORKERS", available_cpus)), available_cpus, 4))
print(f"Competition data: {COMP}")
print(f"Submission tree : {SUB}")
print(f"Workers / CPUs  : {WORKERS} / {available_cpus}")
'''


VERIFY = r'''def load_circuit(path):
    return km.Circuit.from_file(str(path), format="kmb")


def with_heartbeat(label, function, every=60):
    finished = threading.Event()
    started = time.monotonic()

    def heartbeat():
        while not finished.wait(every):
            print(f"    {label} still running ({(time.monotonic() - started) / 60:.1f} min)", flush=True)

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
    lambda: VI.verify(
        load_circuit(SUB / "invsqrt.kmb"),
        VI.ProblemSpecInvSqrt.for_n_p(10),
        max_workers=WORKERS,
    ),
)
assert invsqrt_result["status"] == "PASS" and invsqrt_result["verified_shots"] == 200540928
assert invsqrt_result["num_qubits"] == 82, invsqrt_result

print("[2/3] Full DQI verification", flush=True)
from dqi import verify_submission_dqi as VD

dqi_circuit = load_circuit(SUB / "dqi.kmb")
dqi_trials = []
for salt in ("", "independent-v5-a", "independent-v5-b", "independent-v5-c"):
    result = with_heartbeat(
        f"DQI verifier salt={salt or '<empty>'}",
        lambda salt=salt: VD.verify(
            dqi_circuit,
            VD.ProblemSpecDQI.benchmark_instance(),
            salt=salt,
            max_workers=WORKERS,
        ),
    )
    assert result["status"] == "PASS" and result["verified_shots"] == 102400
    assert result["num_qubits"] == 898, result
    dqi_trials.append({"salt": salt, **result})
    print(
        f"    salt={salt or '<empty>'}: value={result['value_errors']}, "
        f"phase={result['phase_errors']}, R={result['repetition_factor']:.6f}, "
        f"cost={result['score']:.7g}",
        flush=True,
    )
dqi_result = dqi_trials[0]
dqi_worst_result = max(dqi_trials, key=lambda result: result["score"])

print("[3/3] Full DFTHC-SOS verification", flush=True)
from dfthcsos import verify as VC

circuits = [load_circuit(path) for path in sorted(
    (SUB / "dfthcsos").glob("angle_k_*.kmb"),
    key=lambda path: int(path.stem.rsplit("_", 1)[1]),
)]
with np.load(SUB / "dfthcsos" / "submission.npz", allow_pickle=False) as archive:
    factorization = {name: np.array(archive[name]) for name in archive.files}
tree_edges = factorization.pop("tree_edges")
dfthcsos_result = with_heartbeat(
    "chemistry verifier",
    lambda: VC.verify(circuits, factorization, tree_edges),
)
assert dfthcsos_result["status"] == "PASS" and dfthcsos_result["verified_shots"] == 104704
assert dfthcsos_result["num_qubits"] == 656, dfthcsos_result
assert dfthcsos_result["toffoli_count"] == 29360, dfthcsos_result

RESULTS = {
    "invsqrt": invsqrt_result,
    "dqi": dqi_result,
    "dfthcsos": dfthcsos_result,
}
SCORE = math.prod(REFERENCE[name] / RESULTS[name]["score"] for name in REFERENCE) ** (1 / 3)
WORST_SALT_SCORE = (
    (REFERENCE["invsqrt"] / invsqrt_result["score"])
    * (REFERENCE["dqi"] / dqi_worst_result["score"])
    * (REFERENCE["dfthcsos"] / dfthcsos_result["score"])
) ** (1 / 3)
if min(SCORE, WORST_SALT_SCORE) < 10.0:
    raise AssertionError(
        f"Portfolio fell below 10.0: default={SCORE:.6f}, "
        f"worst reviewed salt={WORST_SALT_SCORE:.6f}"
    )

rows = []
for name, label in (("invsqrt", "Inverse square root"), ("dqi", "DQI decoder"),
                    ("dfthcsos", "FeMoCo lookups")):
    result = RESULTS[name]
    rows.append({
        "Task": label,
        "Qubits": result["num_qubits"],
        "Toffoli": f"{result['toffoli_count']:,.3f}",
        "Cost": f"{result['score']:.7g}",
        "Reported cost": f"{PLAN[name]['reported_cost']:.7g}",
        "Reference / cost": f"{REFERENCE[name] / result['score']:.3f}",
    })
display(pd.DataFrame(rows).style.hide(axis="index"))
print(f"Full verification completed in {(time.monotonic() - started) / 60:.1f} min")
print(f"Verified local score: {SCORE:.5f}")
print(f"Worst reviewed-salt score: {WORST_SALT_SCORE:.5f}")
'''


PACKAGE = r'''MAX_ID = 2**30 // 1536
problems = []
all_files = sorted(path.relative_to(SUB).as_posix() for path in SUB.rglob("*") if path.is_file())
angles = [name for name in all_files if name.startswith("dfthcsos/angle_k_") and name.endswith(".kmb")]
expected = {"invsqrt.kmb", "dqi.kmb", "dfthcsos/submission.npz"} | {
    f"dfthcsos/angle_k_{index}.kmb" for index in range(len(angles))
}
if set(all_files) != expected:
    problems.append(
        f"manifest differs: extra={sorted(set(all_files) - expected)}, "
        f"missing={sorted(expected - set(all_files))}"
    )
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
            value.dtype.byteorder == "=" and sys.byteorder == "little"
        )
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
    if len(infos) != len(expected) or {info.filename for info in infos} != expected:
        raise AssertionError("ZIP manifest differs from the verified submission tree")
    if any(info.file_size == 0 for info in infos):
        raise AssertionError("ZIP contains an empty member")
    if archive.testzip() is not None:
        raise AssertionError("ZIP CRC validation failed")
    member_hashes = {
        info.filename: hashlib.sha256(archive.read(info.filename)).hexdigest()
        for info in infos
    }

summary = {
    "score": SCORE,
    "worst_reviewed_salt_score": WORST_SALT_SCORE,
    "results": RESULTS,
    "dqi_salt_trials": dqi_trials,
    "source_plan": PLAN,
    "member_sha256": member_hashes,
    "archive_sha256": sha256_file(ZIP_PATH),
    "archive_bytes": ZIP_PATH.stat().st_size,
}
(WORK / "summary_v5.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
(WORK / "submission-members-v5.sha256").write_text(
    "".join(f"{member_hashes[name]}  {name}\n" for name in sorted(member_hashes))
)
print(f"Submission ready: {ZIP_PATH}")
print(f"Files           : {len(expected)}")
print(f"Archive size    : {ZIP_PATH.stat().st_size / 2**20:.2f} MiB")
print(f"SHA-256         : {summary['archive_sha256']}")
print(f"Verified score  : {SCORE:.5f}")
print(f"Worst salt score: {WORST_SALT_SCORE:.5f}")
ZIP_PATH
'''


REVIEW = r"""## Review findings and next research target

The score gain is distributed across all three tasks. Relative to the verified
v4 costs, the portfolio is about 1.128× better on inverse square root, 1.409×
on DQI and 1.109× on chemistry, for a combined score multiplier near 1.208×.
DQI supplies the largest improvement.

This notebook verifies prebuilt artifacts. The attached study describes the
Q82 fused-phase inverse-square-root compiler, Q898 paired J-fraction extraction
and the `fouraddr2` chemistry loader, but does not include those generators.
Source reproduction is therefore the next review gate. For further score work,
the strongest public numerical inputs remain the R26 group-compressed
factorization (dataset version `21188334`) and the official starter kit
(`21107076`); the older R65 factorization (`21143008`) is useful as a training
seed but is not competitive as a submission.

Artifact code and notebook text are published under Apache-2.0 according to the
supplied study. Numerical datasets retain their own Kaggle dataset terms.
"""


def main() -> None:
    notebook = {
        "cells": [
            cell("markdown", INTRO),
            cell("code", PREFLIGHT),
            cell("code", VERIFY),
            cell("code", PACKAGE),
            cell("markdown", REVIEW),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
            "kaggle": {
                "accelerator": "none",
                "dataSources": [
                    {"sourceType": "competition", "sourceId": 161954},
                    {"sourceType": "datasetVersion", "sourceId": 21216223},
                    {"sourceType": "datasetVersion", "sourceId": 21211427},
                    {"sourceType": "datasetVersion", "sourceId": 21216621},
                ],
                "dockerImageVersionId": 31498,
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()

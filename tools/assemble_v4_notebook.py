#!/usr/bin/env python3
"""Assemble the standalone Kaggle v4 notebook from two reviewed notebooks.

The repository notebook supplies the verified 84-qubit inverse-square-root
builder and dynamic packaging. The public 8.953 notebook supplies the
932-qubit J-fraction decoder and R26 four-address chemistry builder.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def lines(text: str) -> list[str]:
    if text and not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(source),
    }


def assert_layout(current: dict, public: dict) -> None:
    current_markers = {
        35: "%%writefile src/ports/invsqrt_ports.py",
        44: "Building and exactly post-compiling inverse square root",
        51: "ZIP_PATH = P.write_zip",
    }
    public_markers = {
        32: "write_source('dqi/common.py'",
        48: "write_source('dqi/continued_fraction/jfraction_gap_decoder.py'",
        52: "write_source('dfthcsos/group_loader/four_address.py'",
        55: "write_source('tools/check-submission'",
    }
    for index, marker in current_markers.items():
        if marker not in "".join(current["cells"][index]["source"]):
            raise ValueError(f"current notebook cell {index} no longer matches {marker!r}")
    for index, marker in public_markers.items():
        if marker not in "".join(public["cells"][index]["source"]):
            raise ValueError(f"public notebook cell {index} no longer matches {marker!r}")


INTRO = r"""# Verifier-guided hybrid solution: projected score ~9.0

This standalone Kaggle notebook combines the strongest verified public pieces
available in the supplied material:

| Task | Selected construction | Expected resources | Reference / cost |
|---|---|---:|---:|
| inverse square root | our exact-domain Q84 circuit | Q=84, C≈1,854 | ≈13.33× |
| DQI | in-place J-fraction decoder | Q=932, T≈1.424M | ≈16.58× |
| DFTHC-SOS | public R26 factorization + four-address loader | Q=656, T=31,860 | ≈3.29× |

The projected geometric score is about **9.0**. Every run rebuilds all three
components from embedded source, runs the full official verifiers, checks the
submission format, and writes `/kaggle/working/submission.zip`.

## Required Kaggle inputs

1. **Google - Verifiable Quantum Circuit Compilation** competition data.
2. [DF-THC Group Compression Factorization](https://www.kaggle.com/datasets/ahmedberatozer/df-thc-group-compression-factorization), containing the pinned R26 file `dfthc_group_v2_26_60_30.npz`.

Use a CPU notebook with Python 3.13 and select **Restart & Run All**. Internet
and GPU are unnecessary. Allow about 20 minutes and roughly 10 GB peak RAM.

The notebook also runs a verifier-aligned chemistry estimator before circuit
generation. It exposes the `(R,B,C,brot,J)` resource cliffs needed for future
factorization checkpoints without spending a full verifier run on a candidate
that cannot improve the incumbent.
"""


PREFLIGHT = r'''import glob
import hashlib
import json
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time

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
SUB = pathlib.Path(os.environ.get("QCC_SUB", "/tmp/qcc_submission_v4"))
SLUG = "google-verifiable-quantum-circuit-compilation"
R26_SHA256 = "7a38e11a54d95ee5ac99246081a9833270548c9d6ba6c81248763ae664731ee5"
REFERENCE = {
    "invsqrt": 2076774.233715282,
    "dqi": 24439474177519.598,
    "dfthcsos": 3355599588.3155365,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


if sys.version_info[:2] != (3, 13):
    raise RuntimeError(f"kickmix.so requires CPython 3.13; found {platform.python_version()}")
if not INPUT.exists():
    raise FileNotFoundError(f"Kaggle input directory does not exist: {INPUT}")

competition_dirs = competition_candidates(INPUT)
if len(competition_dirs) != 1:
    raise RuntimeError(
        "Expected exactly one competition input containing kickmix/ and verifiers/. "
        f"Found {len(competition_dirs)}: {[str(path) for path in competition_dirs]}"
    )
COMP = competition_dirs[0]

r26_candidates = sorted(INPUT.rglob("dfthc_group_v2_26_60_30.npz"))
r26_hashes = {path: sha256_file(path) for path in r26_candidates}
valid_r26 = [path for path, digest in r26_hashes.items() if digest == R26_SHA256]
if len(valid_r26) != 1:
    raise RuntimeError(
        "Attach Ahmed Berat Ozer's pinned R26 factorization dataset. "
        f"Expected one file with SHA-256 {R26_SHA256}; "
        f"observed {dict(map(lambda item: (str(item[0]), item[1]), r26_hashes.items())) or 'none'}."
    )
R26_DATA = valid_r26[0]
with np.load(R26_DATA, allow_pickle=False) as archive:
    expected = {
        "W": (26, 60, 30),
        "H2_I": (26, 30),
        "angle_words": (1636, 75),
        "tree_edges": (75, 2),
    }
    observed = {name: archive[name].shape for name in expected if name in archive.files}
    if observed != expected:
        raise ValueError(f"Unexpected R26 arrays: {observed}; expected {expected}")

WORK.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)
dfthcsos_out = SUB / "dfthcsos"
dfthcsos_out.mkdir(parents=True, exist_ok=True)
for generated in [SUB / "invsqrt.kmb", SUB / "dqi.kmb", dfthcsos_out / "submission.npz"]:
    generated.unlink(missing_ok=True)
for generated in dfthcsos_out.glob("angle_k_*.kmb"):
    generated.unlink()
unexpected = list(dfthcsos_out.iterdir())
if unexpected:
    raise RuntimeError(f"Refusing to mix unknown files into the submission: {unexpected}")

# Current Q84 source is written under WORK/src by the following notebook cells.
source_dirs = [
    "src",
    "src/gen_dqi_v6",
    "src/gen_dqi_v6/ref",
    "src/gen_invsqrt_r2",
    "src/gen_invsqrt_r3",
    "src/gen_invsqrt_r3/vendor",
    "src/ports",
    "src/ports/evidence_inv",
    "src/ports/public_sources/anazemcev_quantum-circuits-source-exact-compilation/embedded/src",
    "src/ports/public_sources/lightning_v2/embedded",
]
for relative in source_dirs:
    (WORK / relative).mkdir(parents=True, exist_ok=True)

# Public Q932/R26 builders use their own source root and reject chemistry output
# inside it. The separate build directory preserves that contract.
PUBLIC_SESSION = pathlib.Path(tempfile.mkdtemp(prefix="quantum-v4-"))
PUBLIC_ROOT = PUBLIC_SESSION / "source"
PUBLIC_OUT = PUBLIC_SESSION / "build"
(PUBLIC_ROOT / "data").mkdir(parents=True)
PUBLIC_OUT.mkdir(parents=True)
(PUBLIC_ROOT / "data" / "comp").symlink_to(COMP, target_is_directory=True)
(PUBLIC_OUT / "records").mkdir()


def write_source(name, source):
    path = PUBLIC_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


PUBLIC_TIMINGS = []


def public_run(*args, record, isolated=False):
    started = time.monotonic()
    flags = ["-I", "-B"] if isolated else ["-B"]
    result = subprocess.run(
        [sys.executable, *flags, *map(str, args)],
        cwd=PUBLIC_ROOT,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(COMP / "kickmix"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    seconds = time.monotonic() - started
    PUBLIC_TIMINGS.append({"command": list(map(str, args)), "seconds": seconds,
                           "returncode": result.returncode})
    destination = PUBLIC_OUT / "records" / record
    destination.write_text(result.stdout)
    destination.with_suffix(destination.suffix + ".stderr").write_text(result.stderr)
    print(f"{args[0]}: {seconds:.1f} s", flush=True)
    if result.returncode:
        print(result.stdout[-4000:], result.stderr[-4000:], sep="\n")
        raise RuntimeError(f"{args[0]} exited with {result.returncode}")
    return result.stdout


available_cpus = os.cpu_count() or 4
requested_workers = int(os.environ.get("QCC_WORKERS", available_cpus))
WORKERS = max(1, min(requested_workers, available_cpus, 4))
os.chdir(WORK)
print(f"Competition data : {COMP}")
print(f"R26 factorization: {R26_DATA}")
print(f"R26 SHA-256      : {R26_SHA256}")
print(f"Scratch output   : {SUB}")
print(f"Workers / CPUs   : {WORKERS} / {available_cpus}")
'''


BUILD_AND_VERIFY = r'''import shutil
import threading
import time

sys.path.insert(0, str(WORK / "src"))
import qcc_pipeline as P

P.setup(COMP)
sys.path.insert(0, str(WORK / "src" / "ports"))
import invsqrt_ports as IP
import port_pipeline as PP
import kickmix as km

PP.setup(COMP)
started = time.perf_counter()


def with_heartbeat(label, function, every=60):
    finished = threading.Event()
    began = time.perf_counter()

    def heartbeat():
        while not finished.wait(every):
            print(f"    {label} is still running ({(time.perf_counter() - began) / 60:.1f} min)", flush=True)

    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        return function()
    finally:
        finished.set()
        monitor.join()


def load_circuit(path):
    return km.Circuit.from_file(str(path), format="kmb")


print("[1/7] Building and exactly post-compiling inverse square root", flush=True)
EV = WORK / "src" / "ports" / "evidence_inv"
invsqrt_path = IP.build_invsqrt(
    SUB,
    base="r3",
    virtual_bits=1,
    certificate_file=EV / "exact_proof.json",
    calibration_file=EV / "calibration_selected.json",
    tables_file=EV / "virtual_tables.json",
)
baseline_invsqrt = load_circuit(invsqrt_path)
PED = PP.module(
    "post_calibration_exact_domain",
    WORK / "src" / "ports" / "public_sources" / "lightning_v2" / "embedded" / "portable_exact_domain.py",
)
optimized_invsqrt, exact_report = PED.optimize_circuit(baseline_invsqrt, linear=True, mbu=True)
baseline_magic = baseline_invsqrt.operation_statistics().num_toffoli_like
optimized_magic = optimized_invsqrt.operation_statistics().num_toffoli_like
if optimized_invsqrt.num_qubits == baseline_invsqrt.num_qubits and optimized_magic < baseline_magic:
    optimized_invsqrt.to_file(str(invsqrt_path), format="kmb")
    print(f"    exact-domain pass: {baseline_magic:,} -> {optimized_magic:,} static Toffolis; "
          f"{len(exact_report.replacements)} certified replacement(s)", flush=True)
else:
    print("    exact-domain pass found no lower-cost certified circuit; keeping baseline", flush=True)

print("[2/7] Estimating the R26 chemistry resources before circuit generation", flush=True)
estimator = json.loads(public_run(
    "tools/dfthcsos_estimator.py",
    "--R", 26, "--B", 60, "--C", 30, "--brot", 10, "--J", 3,
    "--candidate-npz", R26_DATA,
    "--verifier-root", PUBLIC_ROOT / "data" / "comp",
    "--incumbent-cost", 2342000000,
    "--require-improvement",
    record="dfthcsos.estimate.json",
))
assert estimator["loader"]["toffoli_both_passes"] == 25900, estimator
assert estimator["total"]["toffoli"] == 31860, estimator
assert estimator["total"]["qubits"] == 656, estimator
assert estimator["loader"]["within_128_qubit_cap"], estimator
assert estimator["passes_numerical_acceptance"], estimator
assert estimator["total"]["should_generate_circuits"], estimator
print(f"    projected Q={estimator['total']['qubits']}, T={estimator['total']['toffoli']:,}, "
      f"cost={estimator['total']['cost']:.6g}", flush=True)

print("[3/7] Building the 932-qubit in-place J-fraction DQI decoder", flush=True)
dqi_build = json.loads(public_run(
    "dqi/continued_fraction/jfraction_gap_decoder.py",
    SUB / "dqi.kmb",
    record="dqi.build.json",
))
assert dqi_build["num_qubits"] == 932, dqi_build
assert dqi_build["t_count"] == 0, dqi_build
print(f"    Q={dqi_build['num_qubits']}, static T={dqi_build['toffoli_count']:,}", flush=True)

print("[4/7] Building the pinned R26 four-address chemistry family", flush=True)
chemistry_build_dir = PUBLIC_OUT / "dfthcsos"
public_run(
    "dfthcsos/group_loader/build_dataset.py",
    R26_DATA,
    chemistry_build_dir,
    "--sha256", R26_SHA256,
    record="dfthcsos.build.json",
    isolated=True,
)
shutil.rmtree(SUB / "dfthcsos")
(SUB / "dfthcsos").mkdir()
for path in sorted(chemistry_build_dir.glob("angle_k_*.kmb")) + [chemistry_build_dir / "submission.npz"]:
    shutil.copy2(path, SUB / "dfthcsos" / path.name)
SELECTED_J = 3

print("[5/7] Running the official full-shot inverse-square-root verifier", flush=True)
from invsqrt import verify_submission_invsqrt as VI

invsqrt_result = with_heartbeat(
    "inverse-square-root verifier",
    lambda: VI.verify(
        load_circuit(SUB / "invsqrt.kmb"),
        VI.ProblemSpecInvSqrt.for_n_p(10),
        max_workers=WORKERS,
    ),
)
assert invsqrt_result["status"] == "PASS", invsqrt_result
assert invsqrt_result["verified_shots"] == 200540928, invsqrt_result

print("[6/7] Running the official full-shot DQI verifier", flush=True)
from dqi import verify_submission_dqi as VD

dqi_result = with_heartbeat(
    "DQI verifier",
    lambda: VD.verify(
        load_circuit(SUB / "dqi.kmb"),
        VD.ProblemSpecDQI.benchmark_instance(),
        max_workers=WORKERS,
    ),
)
assert dqi_result["status"] == "PASS", dqi_result
assert dqi_result["verified_shots"] == 102400, dqi_result

print("[7/7] Running the official full-shot chemistry verifier", flush=True)
from dfthcsos import verify as VC

angle_paths = sorted((SUB / "dfthcsos").glob("angle_k_*.kmb"),
                     key=lambda path: int(path.stem.rsplit("_", 1)[1]))
circuits = [load_circuit(path) for path in angle_paths]
with np.load(SUB / "dfthcsos" / "submission.npz", allow_pickle=False) as archive:
    factorization = {key: np.array(archive[key]) for key in archive.files}
tree_edges = factorization.pop("tree_edges")
selected_chemistry = with_heartbeat(
    "chemistry verifier",
    lambda: VC.verify(circuits, factorization, tree_edges),
)
assert selected_chemistry["status"] == "PASS", selected_chemistry
assert selected_chemistry["verified_shots"] == 104704, selected_chemistry
assert selected_chemistry["num_qubits"] == estimator["total"]["qubits"]
assert selected_chemistry["toffoli_count"] == estimator["total"]["toffoli"]

RESULTS = {
    "invsqrt": invsqrt_result,
    "dqi": dqi_result,
    "dfthcsos": selected_chemistry,
}
(WORK / "verification_v4.json").write_text(json.dumps(RESULTS, indent=1, default=str))
print(f"All official verifiers passed in {(time.perf_counter() - started) / 60:.1f} minutes", flush=True)
'''


PACKAGE = r'''import hashlib
import zipfile

check_output = public_run("tools/check-submission", SUB, record="check-submission.txt")
if "ALL CHECKS PASS" not in check_output:
    raise AssertionError(check_output)

ZIP_PATH = P.write_zip(SUB, WORK / "submission.zip")
angle_files = sorted((SUB / "dfthcsos").glob("angle_k_*.kmb"),
                     key=lambda path: int(path.stem.rsplit("_", 1)[1]))
if [int(path.stem.rsplit("_", 1)[1]) for path in angle_files] != list(range(len(angle_files))):
    raise AssertionError("FeMoCo angle circuit indices are not contiguous from zero")
expected_members = {
    "invsqrt.kmb",
    "dqi.kmb",
    "dfthcsos/submission.npz",
    *(f"dfthcsos/{path.name}" for path in angle_files),
}

with zipfile.ZipFile(ZIP_PATH) as archive:
    infos = archive.infolist()
    actual_members = {info.filename for info in infos}
    if len(infos) != len(actual_members):
        raise AssertionError("submission.zip contains duplicate member names")
    if actual_members != expected_members:
        raise AssertionError(
            "Unexpected submission manifest. "
            f"Missing: {sorted(expected_members - actual_members)}; "
            f"extra: {sorted(actual_members - expected_members)}"
        )
    empty_members = [info.filename for info in infos if info.file_size == 0]
    if empty_members:
        raise AssertionError(f"Empty submission members: {empty_members}")
    bad_member = archive.testzip()
    if bad_member is not None:
        raise AssertionError(f"CRC failure in {bad_member}")
    member_hashes = {
        info.filename: hashlib.sha256(archive.read(info.filename)).hexdigest()
        for info in infos
    }

zip_digest = sha256_file(ZIP_PATH)
summary = {
    "verifiers": RESULTS,
    "estimator": estimator,
    "member_sha256": member_hashes,
    "archive_sha256": zip_digest,
    "archive_bytes": ZIP_PATH.stat().st_size,
    "geometric_mean": math.prod(REFERENCE[name] / RESULTS[name]["score"] for name in REFERENCE) ** (1 / 3),
    "public_timings": PUBLIC_TIMINGS,
}
(WORK / "summary_v4.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
(WORK / "submission-members-v4.sha256").write_text(
    "".join(f"{member_hashes[name]}  {name}\n" for name in sorted(member_hashes))
)
print(f"Submission ready: {ZIP_PATH}")
print(f"Selected FeMoCo J: {SELECTED_J}")
print(f"Files           : {len(expected_members)}")
print(f"Archive size    : {ZIP_PATH.stat().st_size / 2**20:.2f} MiB")
print(f"SHA-256         : {zip_digest}")
print(f"Projected score : {summary['geometric_mean']:.5f}")
ZIP_PATH
'''


def assemble(current_path: Path, public_path: Path, output_path: Path) -> None:
    current = json.loads(current_path.read_text())
    public = json.loads(public_path.read_text())
    assert_layout(current, public)

    cells = copy.deepcopy(current["cells"])
    cells[0]["source"] = lines(INTRO)
    cells[1]["source"] = lines(PREFLIGHT)
    cells[44]["source"] = lines(BUILD_AND_VERIFY)
    cells[45]["source"] = lines("""## 1. Inverse square root\n\nThe selected exact-domain circuit keeps the 84-qubit construction from the previous notebook. It remains the strongest of the two audited public designs and is verified over all 200,540,928 shots.\n""")
    cells[46]["source"] = lines("""## 2. DQI decoder\n\nThe decoder now transforms the 840-qubit syndrome bank in place through a J-fraction representation. It peaks at 932 qubits, so the squared-qubit metric beats the lower-gate 1,789-qubit Berlekamp-Massey port. Thirty-five flags repair first-order pivot gaps; the official distribution and failure penalty decide the sampled score.\n""")
    cells[47]["source"] = lines("""## 3. FeMoCo angle lookups\n\nThe chemistry component uses Ahmed Berat Ozer's verified `(R,B,C)=(26,60,30)` factorization and the signed four-address loader at `J=3`. Before building circuits, the notebook independently reproduces the official Q=656 and T=31,860 resource model. The full verifier then checks both 20 mHa residual limits and all 104,704 lookup shots.\n""")
    cells[50]["source"] = lines("""## Sources and scope\n\nThe Q84 inverse-square-root source and provenance are retained from the repository notebook. The Q932 J-fraction decoder and R26 chemistry builder are retained from the supplied public 8.953 notebook, including their Apache-2.0 notices and upstream attribution. The R26 numerical dataset keeps its original competition/dataset terms.\n\nThis version implements the reliable ~9.0 baseline and the cheap chemistry estimator. Lower-rank factorization training and a sub-800k-Toffoli in-place DQI decoder remain research targets; the notebook does not label unverified projections as submissions.\n""")
    cells[51]["source"] = lines(PACKAGE)

    public_sources = copy.deepcopy(public["cells"][32:57])
    estimator_source = (ROOT / "tools" / "dfthcsos_estimator.py").read_text()
    estimator_cell = code_cell(
        "write_source('tools/dfthcsos_estimator.py', " + repr(estimator_source) + ")\n"
    )
    cells = cells[:44] + public_sources + [estimator_cell] + cells[44:]
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    current["cells"] = cells
    current.setdefault("metadata", {})["kaggle"] = {
        "accelerator": "none",
        "dataSources": [
            {"sourceType": "competition", "sourceId": 161954},
            {"sourceType": "datasetVersion", "sourceId": 21188334},
        ],
        "dockerImageVersionId": 31498,
        "isInternetEnabled": False,
        "language": "python",
        "sourceType": "notebook",
        "isGpuEnabled": False,
    }
    output_path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current",
        type=Path,
        default=ROOT / "high_scoring_quantum_circuit_compilation.ipynb",
    )
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "high_scoring_quantum_circuit_compilation_v4.ipynb",
    )
    args = parser.parse_args()
    assemble(args.current.resolve(), args.public.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a guarded v6 portfolio with verifier-aligned chemistry offset refit."""
from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "high_scoring_quantum_circuit_compilation_v5_portfolio.ipynb"
OUTPUT = ROOT / "high_scoring_quantum_circuit_compilation_v6_offset_refit.ipynb"


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


INTRO = r"""# Portfolio v6: exact-metric chemistry offset refit

This experimental revision starts from the fully verified 10.88048 portfolio.
It keeps the Q82 inverse-square-root and Q898 DQI circuits byte-identical and
tries one safe numerical improvement: refit the R26 factorization's `H2_I`
offsets while keeping every decoded angle word and every chemistry circuit
fixed.

The optimizer uses an analytic gradient for `lambda_eff`, searches several
bounded trust regions, and evaluates every finalist with the authoritative
`evaluate_words` and `passes_accuracy` functions. A candidate is selected only
when it improves `lambda_eff` and still satisfies both 20 mHa limits, the strict
Lambda/E_gap floors and the valid spectrum-amplification domain. Otherwise the
notebook retains the exact v5 chemistry NPZ.

After selection it repeats the complete v5 review: all 200,540,928 invsqrt
shots, four independent 102,400-shot DQI trials, all 104,704 chemistry shots,
the extra metric-format checks and ZIP CRC validation.

This refit tests the remaining safe route toward a modest score-11-class
improvement. Reaching the 808k DQI or inner-QROAM-cliff targets still requires
circuit/trainer source that is not present in the public artifact datasets.
"""


def optimizer_cell(source: str) -> str:
    return f'''# Embed and test the reviewed fixed-word offset optimizer.
REFIT_DIR = WORK / "v6_refit_source"
REFIT_DIR.mkdir(parents=True, exist_ok=True)
(REFIT_DIR / "dfthcsos_offset_refit.py").write_text({source!r})
sys.path.insert(0, str(REFIT_DIR))
import dfthcsos_offset_refit as OR

gradient_error = OR.gradient_check()
assert gradient_error < 2e-5, gradient_error
print(f"Analytic-gradient check: {{gradient_error:.3e}}")

from dfthcsos import verify as VC


def load_chemistry_circuit(path):
    return km.Circuit.from_file(str(path), format="kmb")


chemistry_paths = sorted(
    (SUB / "dfthcsos").glob("angle_k_*.kmb"),
    key=lambda path: int(path.stem.rsplit("_", 1)[1]),
)
chemistry_circuits = [load_chemistry_circuit(path) for path in chemistry_paths]
with np.load(SUB / "dfthcsos" / "submission.npz", allow_pickle=False) as archive:
    chemistry_archive = {{name: np.array(archive[name]) for name in archive.files}}
tree_edges_for_refit = np.array(chemistry_archive["tree_edges"])

print("Recovering the fixed decoded words with one shot per address", flush=True)
baseline_inputs = VC.load_inputs(chemistry_archive)
quick_evaluation = VC.evaluate_circuits(
    baseline_inputs,
    chemistry_circuits,
    tree_edges_for_refit,
    shots_per_address=1,
    seed=20261025,
)
angle_words_for_refit = np.array(quick_evaluation["angle_words"])
brot_for_refit = int(quick_evaluation["brot"])

print("Searching bounded H2_I/offset refits", flush=True)
selected_factorization, OFFSET_REFIT_REPORT = OR.refit_offsets(
    chemistry_archive,
    angle_words_for_refit,
    tree_edges_for_refit,
    brot_for_refit,
)
baseline_lambda = float(OFFSET_REFIT_REPORT["baseline_metrics"]["lambda_eff"])
selected_lambda = float(OFFSET_REFIT_REPORT["selected_metrics"]["lambda_eff"])

if OFFSET_REFIT_REPORT["improved"]:
    payload = {{name: np.array(value) for name, value in selected_factorization.items()}}
    payload["tree_edges"] = tree_edges_for_refit
    payload["angle_words"] = angle_words_for_refit
    temporary_npz = SUB / "dfthcsos" / "submission.refit.npz"
    np.savez(temporary_npz, **payload)
    temporary_npz.replace(SUB / "dfthcsos" / "submission.npz")
    print(f"Selected lambda_eff {{baseline_lambda:.9f}} -> {{selected_lambda:.9f}}", flush=True)
else:
    print(f"No accepted improvement below lambda_eff={{baseline_lambda:.9f}}; retaining v5", flush=True)

for trial in OFFSET_REFIT_REPORT["trials"]:
    metrics = trial["metrics"]
    print(
        f"radius={{trial['radius']:.1e}} accepted={{trial['accepted']}} "
        f"lambda={{metrics['lambda_eff']:.9f}} "
        f"e1={{1e3 * metrics['error_1']:.4f}}mHa "
        f"e2={{1e3 * metrics['error_2']:.4f}}mHa",
        flush=True,
    )
(WORK / "offset_refit_v6.json").write_text(
    json.dumps(OFFSET_REFIT_REPORT, indent=2, default=str) + "\\n"
)
del baseline_inputs, quick_evaluation, selected_factorization
del chemistry_archive, chemistry_circuits, angle_words_for_refit
import gc
gc.collect()
'''


REVIEW = r"""## What this revision can and cannot change

`H2_I` is read directly from `submission.npz`; changing it does not require
recompiling the fixed ANGLE circuits. This makes the offset refit a genuine,
reviewable numerical optimization rather than a projected circuit count. The
full verifier remains the final authority.

The DQI and invsqrt artifacts are compiled binaries. Their missing EXP-119a
and EXP-225a generators prevent safe gate-level changes here. The notebook
therefore preserves their verified bytes. The next large revision needs either
the paired J-fraction generator for a <=900k-Toffoli DQI build or starter-kit
training that produces a quantized `(R,B,C)` checkpoint across the inner-QROAM
block-16 transition.
"""


def main() -> None:
    notebook = json.loads(BASE.read_text())
    cells = copy.deepcopy(notebook["cells"])
    if len(cells) != 5 or "Verifier-first score-10 portfolio" not in "".join(cells[0]["source"]):
        raise ValueError("v5 notebook layout changed")

    cells[0]["source"] = lines(INTRO)
    source = (ROOT / "tools" / "dfthcsos_offset_refit.py").read_text()
    cells.insert(2, code_cell(optimizer_cell(source)))

    verify_source = "".join(cells[3]["source"])
    verify_source = verify_source.replace(
        "if min(SCORE, WORST_SALT_SCORE) < 10.0:",
        "if min(SCORE, WORST_SALT_SCORE) < 10.8:",
    ).replace(
        'f"Portfolio fell below 10.0: default={SCORE:.6f}, "',
        'f"Portfolio fell below the v6 safety floor 10.8: default={SCORE:.6f}, "',
    )
    cells[3]["source"] = lines(verify_source)

    package_source = "".join(cells[4]["source"])
    package_source = package_source.replace(
        '"source_plan": PLAN,',
        '"source_plan": PLAN,\n    "offset_refit": OFFSET_REFIT_REPORT,',
    ).replace("summary_v5.json", "summary_v6.json").replace(
        "submission-members-v5.sha256", "submission-members-v6.sha256"
    )
    cells[4]["source"] = lines(package_source)
    cells[5]["source"] = lines(REVIEW)

    for item in cells:
        if item["cell_type"] == "code":
            item["execution_count"] = None
            item["outputs"] = []
    notebook["cells"] = cells
    OUTPUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()

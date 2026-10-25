# Verifiable Quantum Circuit Compilation notebook

[`high_scoring_quantum_circuit_compilation_v6_offset_refit.ipynb`](high_scoring_quantum_circuit_compilation_v6_offset_refit.ipynb) is the next experimental revision. It starts from the verified v5 artifacts, performs a bounded fixed-word chemistry offset refit with an analytic gradient, selects candidates only through the official exact metrics, and falls back to the v5 factorization when no accepted improvement exists. Run v6 before submitting it; its numerical gain depends on the attached R26 checkpoint.

[`high_scoring_quantum_circuit_compilation_v5_portfolio.ipynb`](high_scoring_quantum_circuit_compilation_v5_portfolio.ipynb) is the recommended submission notebook. Its recorded Kaggle review run passed every full-shot verifier and four independent DQI salts, scoring **10.88048** on both the default and worst reviewed salt. It pins the three public artifact datasets by exact hash, enforces the metric's extra format rules, and packages only if the verified score remains above 10. See [`SCORE_10_REVIEW.md`](SCORE_10_REVIEW.md) for the audit and dataset inventory.

[`high_scoring_quantum_circuit_compilation_v4.ipynb`](high_scoring_quantum_circuit_compilation_v4.ipynb) is the source-embedded fallback. It combines the 84-qubit inverse-square-root circuit with the public 932-qubit J-fraction DQI decoder and the verified R26 four-address chemistry construction, for a score near 9.0.

Add these Kaggle inputs before selecting **Restart & Run All**:

1. The Google Verifiable Quantum Circuit Compilation competition data.
2. [DF-THC Group Compression Factorization](https://www.kaggle.com/datasets/ahmedberatozer/df-thc-group-compression-factorization), containing `dfthc_group_v2_26_60_30.npz`.

Use Kaggle's Python 3.13 CPU image. The notebook runs offline, validates the pinned R26 file by SHA-256, builds every circuit from embedded source, runs all three official verifiers at their full shot counts, checks the submission manifest and CRCs, and writes `/kaggle/working/submission.zip`. Expect about 20 minutes and roughly 10 GB peak RAM.

[`tools/dfthcsos_estimator.py`](tools/dfthcsos_estimator.py) reproduces the official state-preparation model and the public four-address loader schedule before circuit generation. It reports address width, the 128-qubit loader limit, Q/T totals, and exact quantized factor metrics when a candidate NPZ and the full competition data are available.

The earlier [`high_scoring_quantum_circuit_compilation.ipynb`](high_scoring_quantum_circuit_compilation.ipynb) remains as the fully source-built 6.45-series baseline. [`SCORE_GAP_ANALYSIS.md`](SCORE_GAP_ANALYSIS.md) explains why lower-rank chemistry training and a lower-Toffoli in-place DQI construction are the remaining paths toward double-digit scores.

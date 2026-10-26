# Verifiable Quantum Circuit Compilation notebook

[`high_scoring_quantum_circuit_compilation_v7_qroam_search.ipynb`](high_scoring_quantum_circuit_compilation_v7_qroam_search.ipynb) is the current improvement notebook. It responds to the measured v6 score of **10.88057** by changing chemistry dimensions instead of continuing the exhausted offset-only search. It contribution-prunes and retrains several `(R,B,C)` shapes, quantizes before the final weight refit, builds circuits only after the official numerical gates pass, and replaces the incumbent only when exact circuit resources are cheaper. The default full search includes the R19/B49/C24 address transition and R12/B36/C20 inner-QROAM transition. Every failure path retains the verified score-10 portfolio.

Run v7 on a Kaggle T4 with these inputs, which are already pinned in its notebook metadata:

1. The competition data.
2. `qcc-best-invsqrt-circuit`, dataset version `21216223`.
3. `qcc-best-dqi`, dataset version `21211427`.
4. `qcc-best-circuits`, dataset version `21216621`.
5. The official Verifiable Circuit Compilation Starter Kit, dataset version `21107076`.

The default `TRAIN_PRESET="full"` is a long research run. Set it to `"score11"` in the training cell to stop after the near-frontier R26 and R20 branches. Both modes write a partial report and continuous checkpoints after each branch. Only the final full-verifier table and `summary_v7.json` are submission evidence.

[`high_scoring_quantum_circuit_compilation_v6_offset_refit.ipynb`](high_scoring_quantum_circuit_compilation_v6_offset_refit.ipynb) is the completed offset-only experiment. Its Kaggle result, 10.88057, shows that fixed-word `H2_I` optimization has negligible remaining leverage.

[`high_scoring_quantum_circuit_compilation_v5_portfolio.ipynb`](high_scoring_quantum_circuit_compilation_v5_portfolio.ipynb) is the recommended submission notebook. Its recorded Kaggle review run passed every full-shot verifier and four independent DQI salts, scoring **10.88048** on both the default and worst reviewed salt. It pins the three public artifact datasets by exact hash, enforces the metric's extra format rules, and packages only if the verified score remains above 10. See [`SCORE_10_REVIEW.md`](SCORE_10_REVIEW.md) for the audit and dataset inventory.

[`high_scoring_quantum_circuit_compilation_v4.ipynb`](high_scoring_quantum_circuit_compilation_v4.ipynb) is the source-embedded fallback. It combines the 84-qubit inverse-square-root circuit with the public 932-qubit J-fraction DQI decoder and the verified R26 four-address chemistry construction, for a score near 9.0.

The source-built v4 needs these Kaggle inputs before selecting **Restart & Run All**:

1. The Google Verifiable Quantum Circuit Compilation competition data.
2. [DF-THC Group Compression Factorization](https://www.kaggle.com/datasets/ahmedberatozer/df-thc-group-compression-factorization), containing `dfthc_group_v2_26_60_30.npz`.

Use Kaggle's Python 3.13 CPU image. The notebook runs offline, validates the pinned R26 file by SHA-256, builds every circuit from embedded source, runs all three official verifiers at their full shot counts, checks the submission manifest and CRCs, and writes `/kaggle/working/submission.zip`. Expect about 20 minutes and roughly 10 GB peak RAM.

[`tools/dfthcsos_estimator.py`](tools/dfthcsos_estimator.py) reproduces the official state-preparation model and the public four-address loader schedule before circuit generation. It reports address width, the 128-qubit loader limit, Q/T totals, and exact quantized factor metrics when a candidate NPZ and the full competition data are available.

[`tools/dfthcsos_dimension_train.py`](tools/dfthcsos_dimension_train.py) contains the deterministic contribution ranking and strict candidate-selection logic used by v7. [`tools/dfthcsos_four_address.py`](tools/dfthcsos_four_address.py) is the reviewed Apache-2.0 public four-address compiler used to price and build every new shape.

The earlier [`high_scoring_quantum_circuit_compilation.ipynb`](high_scoring_quantum_circuit_compilation.ipynb) remains as the fully source-built 6.45-series baseline. [`SCORE_GAP_ANALYSIS.md`](SCORE_GAP_ANALYSIS.md) explains why lower-rank chemistry training and a lower-Toffoli in-place DQI construction are the remaining paths toward double-digit scores.

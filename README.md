# Verifiable Quantum Circuit Compilation notebook

`high_scoring_quantum_circuit_compilation.ipynb` is a Kaggle-ready, source-built submission for all three competition tasks. It retains the validated circuit algorithms from the supplied reference notebook and adds deterministic input discovery, an exhaustive inverse-square-root compiler pass, FeMoCo gauge and batch-size optimization, safe reruns, full official verification, one-minute progress heartbeats, and archive-manifest checks.

Add these Kaggle inputs before selecting **Restart & Run All**:

1. The Google Verifiable Quantum Circuit Compilation competition data.
2. [VQCC FeMoCo public factorization, version 2](https://www.kaggle.com/datasets/dmitriigluzdov/vqcc-femoco-public-factorization/versions/2).

The final artifact is `/kaggle/working/submission.zip`. The recorded v1 source run passed all official verifiers and produced a geometric score of 6.434. This revision searches verifier-approved improvements during the run. DQI and measurement-based lookup tests are salted on the leaderboard, so exact scores can vary slightly.

The notebook runs offline and carries source-level attribution and Apache-2.0 licensing details in its final section.

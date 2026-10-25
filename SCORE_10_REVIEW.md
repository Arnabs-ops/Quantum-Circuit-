# Score-10 portfolio review

Date: 2026-10-25

## Decision

The supplied **QCC | Verified Portfolio + DF-THC Retraining Study** is the
strongest submission path found so far. Its reported costs produce a geometric
score of **10.863873**. The arithmetic and its comparison with our verified v4
are internally consistent.

| Task | v4 verified cost | Portfolio reported cost | Cost improvement | Portfolio factor |
|---|---:|---:|---:|---:|
| invsqrt | 155,756.409 | 138,095.135 | 1.1279× | 15.0387× |
| DQI | 1.473966e12 | 1.045962e12 | 1.4092× | 23.3655× |
| DFTHC-SOS | 1.020072e9 | 9.196081e8 | 1.1092× | 3.6489× |

The combined improvement is 1.2081×, moving the v4 estimate from 8.99281 to
10.86387. DQI provides the largest gain.

This result is an artifact portfolio rather than a source reproduction. Its
only code cell locates three prebuilt datasets, checks their hashes and writes
the ZIP. The claims about multiple salts and source-level changes refer to
separate experiments that are not embedded in the supplied notebook.

## Exact Kaggle artifacts

The notebook metadata pins the competition and three dataset-version IDs. Its
packaging plan supplies the matching slugs and content hashes.

| Role | Dataset slug | Version ID | Expected content hash |
|---|---|---:|---|
| Q82 invsqrt | `qcc-best-invsqrt-circuit` | `21216223` | `0f84f5bab3a02e6b357d4656553607174345a33d252cef643f9136db13212141` |
| Q898 DQI | `qcc-best-dqi` | `21211427` | `c5da9cd4e2d73dd41417e539f2659009d646b3ee30ddb5ca494261378a502856` |
| R26 chemistry family | `qcc-best-circuits` | `21216621` | `47187a5d3664c76e48f1266c39a24b55a7bcc845743d6a673c26c8e69b5cbcee` |

The chemistry hash covers the relative path and bytes of every file in the
dataset directory. The final submission contains 28 files: invsqrt, DQI,
25 angle circuits and `submission.npz`.

## What changed relative to v4

### Inverse square root

The study reports Q84 to Q82 and about 1,854 to 1,684 Toffoli-equivalents.
It attributes the reduction to carry-save outer rows, removal of one derivative
wire, a one-bit normalization ladder and a fused phase stage. The reported cost
is 138,095.135. The uploaded notebook does not contain the EXP-225a generator,
so the Q82 artifact can be verified but not rebuilt from this material.

### DQI

The study starts from the public Q932 J-fraction decoder, packs gap state and
releases an extraction buffer to reach Q898, then extracts two continued-
fraction levels per reciprocal. It reports about 1.08M mean Toffolis and cost
1.045962e12. This is the main score gain. The paired extraction generator is
not present, so full-shot verification across additional salts is the critical
review gate.

### Chemistry

The numerical factorization remains R26. The study reports a fixed-word weight
refit that lowers `lambda_eff`, followed by a `fouraddr2` loader that reduces
angle Toffolis from 25,900 to 23,400. With state preparation and rotations,
T_total becomes 29,360 at Q=656 and the reported cost is 919,608,141. The
generator and refit checkpoint are absent from the uploaded notebook.

## Dataset inventory for further work

These exact dataset versions are present in the supplied notebook metadata and
earlier reviewed notebooks:

| Version ID | Dataset/use | Review value |
|---:|---|---|
| `21216223` | best Q82 invsqrt artifact | immediate score-10 submission |
| `21211427` | best Q898 DQI artifact | immediate score-10 submission; highest salt risk |
| `21216621` | best R26 chemistry artifacts | immediate score-10 submission |
| `21188334` | Ahmed Berat Ozer R26/R32 group-compressed factorization | best public numerical seed for chemistry refitting |
| `21107076` | official Verifiable Circuit Compilation Starter Kit | authoritative trainer/build scaffolding |
| `21143008` | Dmitrii Gluzdov R65 public factorization v2 | useful training/provenance seed; inferior submission cost |

The three score-10 artifact datasets should be used first. The R26 dataset and
starter kit are the useful pair for source reproduction and future numerical
training. R65 is valuable for studying group pruning but its scored resources
are no longer competitive.

Direct Kaggle API calls from this execution environment were rejected by the
network proxy with HTTP 403, including after network permission was granted.
The IDs above come from the notebooks' Kaggle `dataSources` metadata rather
than name-only web-search results, so they are exact inputs used by the recorded
runs. Dataset contents still require a Kaggle execution for independent byte
and verifier checks.

## Implemented review notebook

`high_scoring_quantum_circuit_compilation_v5_portfolio.ipynb` hardens the
supplied packager:

1. attaches the exact three dataset versions and competition data;
2. requires one unambiguous directory for each dataset slug;
3. checks all three published content hashes before copying anything;
4. copies only the 28 allowed submission members;
5. runs 200,540,928 invsqrt shots, four independent 102,400-shot DQI trials
   and 104,704 chemistry shots with the official verifiers;
6. checks Q82, Q898, Q656 and chemistry T=29,360;
7. enforces the metric's register-name, largest-ID, NPZ dtype and manifest
   rules, tests ZIP CRCs and records every member hash;
8. refuses to package if either the default or worst reviewed-salt geometric
   score is below 10.

The notebook is statically validated locally. The prebuilt datasets are not
available in this workspace, so its dynamic review must run on Kaggle before it
replaces v4 as the recommended submission.

## Next review gates

1. Run the v5 notebook on Kaggle and retain `summary_v5.json`, the three
   verifier results and `submission-members-v5.sha256`.
2. Extend the notebook's four full-shot DQI salts with one 409,600-shot run.
   Record value, phase and cleanup failures separately. This tests the claimed
   two-gap approximation beyond the default seed.
3. Inspect any provenance files shipped beside the three artifacts and locate
   the EXP-225a, EXP-119a and EXP-231c generators. Byte-for-byte rebuilds are
   required before calling the portfolio source-reproducible.
4. Re-run the fixed-word chemistry metrics independently, comparing `W`,
   `H2_I`, `angle_words`, `Lambda`, `E_gap`, errors and `lambda_eff` against the
   public R26 checkpoint. This separates numerical refit gains from loader gains.
5. If the artifacts and salts pass, submit v5. Further research should focus on
   DQI because it delivered about 60% of the logarithmic score improvement from v4;
   chemistry source reproduction is the second priority.

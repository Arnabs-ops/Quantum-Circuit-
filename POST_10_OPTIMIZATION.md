# Optimization plan after the 10.88048 portfolio

Date: 2026-10-25

## Current measured point

| Task | Q | Mean Toffoli / T-equivalent | Cost | Factor |
|---|---:|---:|---:|---:|
| invsqrt | 82 | 1,365 Toffoli; 1,684.09 equivalent | 138,095.1 | 15.039× |
| DQI | 898 | 1,083,843.277 | 1.04118e12 | 23.473× |
| DFTHC-SOS | 656 | 29,360 | 9.196081e8 | 3.649× |

The chemistry cost implies `lambda_eff = 47.74665`. The DQI result is at the
zero-failure statistical floor, `R = 1.19126`, so more salt testing cannot lower
its expected score materially. Further improvement requires smaller circuits or
a better chemistry factorization.

## Score targets

Because the final score is a cube root, moving from 10.88048 requires these
product improvements:

| Target score | Required product improvement | One-task cost reduction if isolated |
|---:|---:|---:|
| 11 | 1.0333× | 3.2% |
| 12 | 1.3415× | 25.5% |
| 13 | 1.7056× | 41.4% |
| 15 | 2.6202× | 61.8% |
| 17 | 3.8142× | 73.8% |

At fixed Q and repetition factor, DQI alone reaches 12 at about **808k mean
Toffolis** and 13 at about **635k**. At fixed chemistry Q and T, chemistry alone
reaches 12 at `lambda_eff ≈ 35.6`; alternatively it needs T_total about 21.9k at
the current lambda. Those are major algorithmic changes.

## Priority 1: chemistry dimensions and the QROAM cliff

Chemistry remains the weakest task factor. Circuit-only cleanup on the current
R26 checkpoint has limited leverage: the 23,400 angle Toffolis are already only
468 per circuit per pass. Even eliminating another 20 gates from every pass
saves about 1,000 total Toffolis, roughly 3.4% of T_total and only about 1.1% on
the final score.

The large opportunity is the verifier's inner-QROAM workspace transition. The
resource estimator gives the following projections at the current
`lambda_eff = 47.74665`. These use the older public four-address schedule, so
they are conservative relative to `fouraddr2`; they are resource projections,
not accepted Hamiltonian fits.

| `(R,B,C)` | Rows `76+R*B` | Address bits | Inner block | Q | T | Projected chemistry factor |
|---|---:|---:|---:|---:|---:|---:|
| (26,60,30) | 1,636 | 11 | 32 | 656 | 31,860 | 3.36× |
| (19,49,24) | 1,007 | 10 | 32 | 655 | 22,873 | 4.69× |
| (14,60,24) | 916 | 10 | 32 | 652 | 21,409 | 5.03× |
| (12,60,24) | 796 | 10 | 32 | 652 | 19,774 | 5.45× |
| (12,36,20) | 508 | 9 | 16 | 380 | 15,859 | 11.66× |
| (12,30,16) | 436 | 9 | 16 | 361 | 14,547 | 13.38× |

The important boundary is not rank alone. A candidate around `R=12`, `B<=36`,
`C<=20` halves the selected inner block and removes roughly 270 billed qubits.
If such a quantized factorization retained lambda near 48 and passed both 20 mHa
limits, the current invsqrt and DQI circuits would score around 16. That is the
credible path from 10 to the upper leaderboard range.

The numerical work should proceed in this order:

1. Start from the public R26 words and weights, plus the starter-kit trainer.
2. Prune whole groups and basis vectors with the smallest reconstruction
   contribution, refitting after each removal rather than truncating once.
3. Reduce C jointly with R/B. The QROAM transition requires C reduction; rank
   reduction alone leaves the 541-qubit temporary workspace almost unchanged.
4. Optimize the exact verifier objective, including `lambda_eff`, error_1,
   error_2, the strict Lambda/E_gap floors and the quantized decoded vectors.
5. Quantize angles before the final weight refit. Use the fixed decoded vectors
   when refitting `W` and `H2_I`, then spend the remaining error budget up to
   about 19.9 mHa on lower lambda.
6. Reject candidates with the estimator before building ANGLE circuits.

## Priority 2: DQI below 900k Toffolis

The Q898 circuit is already close to the compulsory 840 input qubits. Removing
20 more qubits saves only 4.4% of Q-squared cost. The main DQI target is gates:

| Mean Toffolis at Q898 | DQI factor | Overall score with current other tasks |
|---:|---:|---:|
| 1,083,843 | 23.47× | 10.88 |
| 950,000 | 26.78× | 11.37 |
| 900,000 | 28.27× | 11.58 |
| 808,000 | 31.49× | 12.00 |
| 700,000 | 36.35× | 12.58 |

The supplied artifact does not include the EXP-119a generator, so binary-level
rewrites are not a productive route. The next gate is obtaining or reproducing
the paired J-fraction extraction source. Once available, profile the reciprocal,
tail-division and Forney scan stages independently. Preserve the two-gap failure
budget and test changes with millions of classical samples before full kickmix
runs. The target should be Q<=900 and T<=900k first, then T<=808k.

## Priority 3: inverse square root

Inversqrt is already the strongest factor per implementation effort. A useful
next point would be Q80 with about 1,550 Toffoli-equivalents, cost 124k and a
factor near 16.7. That raises the final score only about 3.6%. It should follow
chemistry training and DQI source recovery.

## Practical next experiment

Use the verified v5 portfolio as the submission base and create a chemistry
training notebook with four attached inputs: competition data, starter kit
`21107076`, public R26/R32 factors `21188334`, and best chemistry artifacts
`21216621`. Train and checkpoint candidates around these boundaries:

- `(R,B,C)=(19,49,24)` to cross the 1,024-row address boundary;
- `(14,60,24)` and `(12,60,24)` to test group pruning while retaining B;
- `(12,36,20)` to target the QROAM block-16 transition;
- `(12,30,16)` as the high-upside target.

Each checkpoint should store continuous factors, quantized words, exact metrics,
estimator Q/T/cost, optimizer seed and parent checkpoint. Only candidates that
pass the numerical verifier should proceed to circuit generation and full-shot
verification.

## Implemented v6 experiment

`high_scoring_quantum_circuit_compilation_v6_offset_refit.ipynb` implements the
safe optimization available without the missing circuit generators. It keeps
all R26 words and circuits fixed and optimizes only the submitted `H2_I`
offsets. The objective has an analytic gradient through the eigenvalue and LCU
expressions; a finite-difference regression matches it within `2.5e-9`.

Four bounded trust regions are searched. Every result is re-evaluated by the
official exact metric, and only an accepted lower-`lambda_eff` candidate
replaces the v5 NPZ. The notebook then reruns the complete full-shot portfolio
review and retains v5 automatically when the refit does not improve it.

This implements a real chemistry change that can be validated with the current
public artifacts. The 808k DQI and block-16 QROAM targets remain generator and
training problems; compiled artifact bytes alone do not provide a safe route to
those structural changes.

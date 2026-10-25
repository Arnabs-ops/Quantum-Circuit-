# Verifiable Quantum Circuit Compilation: score-gap analysis

Date: 2026-10-25

## Finding

The 6.451 notebook is not missing a small compiler flag. It is using the wrong
global trade-offs for the two benchmarks that still have large headroom:

1. **DQI keeps too much state live.** It uses 1,789 qubits. Because the metric
   is `R * Q^2 * T`, its low 445k Toffoli count does not compensate for the
   workspace. The attached 8.95 notebook deliberately spends 3.2 times as many
   Toffolis to reduce the footprint to 932 qubits, and wins.
2. **Chemistry never changes the factorization dimensions.** The 6.451 notebook
   refines a gauge of a fixed `(R,B,C)=(65,60,30)` factorization. The competition
   paper makes `(R,B,C)` part of the optimization variable, and its outer
   objective is the actual block-encoding cost times `lambda_eff`. The attached
   public notebook drops and refits the 65 groups to `R=26`; this more than
   halves the chemistry cost. Scores in the 17-20 range require going beyond
   that public R26 checkpoint and improving DQI at the same time.

The inverse-square-root circuit is not the present bottleneck. Our 84-qubit
version has a factor of about 13.33, slightly better than the attached public
notebook's 13.157.

## Measured comparison

| Task | Current 6.451 notebook | Attached public 8.953 notebook | Consequence |
|---|---:|---:|---|
| invsqrt | Q=84, C~1,854, factor 13.33 | Q=90, C=1,753.80, factor 13.157 | Current version is about 1.3% better by task cost. |
| DQI | Q=1,789, T=445,410, R=1.220, factor 14.05 | Q=932, T=1,424,456, R=1.191, factor 16.581 | Squaring Q makes the 932-qubit circuit about 15% cheaper despite 3.2x more gates. |
| DFTHC-SOS | R65, Q=681, T=72,404, lambda=47.51, factor 1.43 | R26, Q=656, T=31,860, lambda=48.807, factor 3.290 | Fixed rank is the largest demonstrated loss. The R26 circuit is about 56% cheaper even with a slightly worse lambda. |

Replacing current DQI and chemistry with the public constructions while
keeping our inverse square root projects to a score of about **9.0**. That is a
useful next submission, but it does not explain a score of 20.

## What a score of 20 requires

The final score is the cube root of the product of the three task-improvement
factors. A score of 20 therefore needs a product of `20^3 = 8000`. The public
8.953 solution has a product of about 718, so it needs another **11.15x in the
product**, spread across one or more tasks.

With the current inverse-square-root factor of 13.33:

| DQI factor | Chemistry factor | Projected overall score |
|---:|---:|---:|
| 16.58 | 3.29 | 8.99 |
| 30 | 12 | 16.87 |
| 40 | 12 | 18.56 |
| 35 | 15 | 19.13 |
| 30 | 20 | 20.00 |
| 40 | 20 | 22.01 |

This is why a top score is unlikely to come from one more inverse-square-root
rewrite. The plausible route is a simultaneous DQI space-time improvement and
a much smaller, implementation-aware chemistry factorization.

## DQI: the missing hybrid

The current notebook ports the truncated Berlekamp-Massey construction from
Gu et al. It is gate-efficient but workspace-heavy: the public source reports
442,397 Toffolis and 1,837 qubits; our port is 445,410 mean Toffolis and 1,789
qubits.

The attached 8.95 notebook uses a different representation:

- It transforms the 70 syndrome words in place into a J-fraction/continued
  fraction and then into numerator/denominator polynomials.
- It reuses the 840 input qubits and peaks at 932 qubits, only 92 above the
  compulsory input size.
- Thirty-five flags repair first-order zero pivots. Unsupported higher-order
  gaps are permitted to fail under the benchmark's sampled failure budget.
- Its cost is 1,424,456 mean Toffolis. An included block-shear cost model
  projects 1,262,247 Toffolis at about 936 qubits, only a modest factor-18.7
  DQI result.

The research target is to combine these strengths: use an in-place
continued-fraction or in-place key-equation representation, but retain the
42-Toffoli field multiplication, 66-Toffoli inverse, derivative-free phase
scan, fixed support windows, and phase-only Forney path from the Gu design.

At repetition factor 1.2, the Toffoli ceilings at Q=932 are:

| Desired DQI factor | Maximum mean Toffolis |
|---:|---:|
| 20 | 1,172,327 |
| 30 | 781,551 |
| 40 | 586,163 |
| 50 | 468,931 |

Factor 30 requires a 45% gate reduction from the public in-place decoder.
Factor 50 roughly requires our present gate count at the public qubit count.
That is difficult, but it is the correct optimization target. Reducing the
current 1,789-qubit decoder by tens of qubits is not competitive because Q is
squared.

The metric also rewards controlled approximation. Zero observed failures still
pays a statistical floor near `R=1.19`; a true failure rate around `1e-3`
raises R only to about 1.21. Fixed windows and limited gap repair can therefore
save meaningful resources when the full 102,400-shot verifier remains below
the failure trade-off.

## Chemistry: rank and verifier cliffs dominate batching

The current notebook searches the angle batch size J and applies orthogonal
gauge rotations, but preserves the augmented Gram matrix, the angle dictionary,
and `(65,60,30)`. It therefore cannot discover the primary savings allowed by
paper equations (62)-(63): changing `(R,B,C)`, refitting the numerical factors,
and minimizing the implementation-aware objective.

The attached public notebook provides direct evidence. Its R26 checkpoint was
trained by starting from 65 groups, dropping groups, refitting the remainder,
and finally fixing ten-bit angles. Its verified metrics are:

- `(R,B,C)=(26,60,30)`
- one-body error 13.476 mHa and two-body error 19.838 mHa
- `Lambda=293.6874`, `E_gap=4.08392`, `lambda_eff=48.80691`
- Q=656, T=31,860, task factor 3.2896

The loader and the analytical state-preparation model contain discrete cliffs:

- The angle address width is `ceil(log2(76 + R*B))`.
- The verifier's inner QROAM changes its chosen block size discontinuously as
  `R*C*B`, B, and the available rotation ancilla change.
- Temporary workspace is shared by taking a maximum, so crossing a QROAM
  threshold can remove hundreds of billed qubits at once.
- Rotation precision changes both the persistent angle registers and 300
  Toffolis per bit across the forward and reverse Givens rotations.

I ran the attached public four-address loader and the official analytical cost
model on illustrative shapes. The R26 row exactly reproduces its reported
25,900 angle Toffolis. The smaller rows use dense, valid signed words to measure
the compiler resource schedule; they are resource projections, not accepted
Hamiltonian factorizations.

| `(R,B,C)` | Addresses | Address bits | Projected Q | Projected T | Chemistry factor if lambda=35 | if lambda=25 |
|---|---:|---:|---:|---:|---:|---:|
| (26,60,30) | 1,636 | 11 | 656 | 31,860 | 4.59 | 6.42 |
| (20,50,24) | 1,076 | 11 | 655 | 24,088 | 6.08 | 8.51 |
| (16,40,20) | 716 | 10 | 652 | 18,824 | 7.81 | 10.94 |
| (12,30,16) | 436 | 9 | 361 | 14,547 | 18.26 | 25.56 |
| (8,24,12) | 268 | 9 | 357 | 12,152 | 22.10 | 30.94 |

The large jump between `(16,40,20)` and `(12,30,16)` is the verifier's QROAM
workspace cliff. This table explains how a private checkpoint with a much more
rank-efficient fit could produce a chemistry factor in the teens or twenties.
Whether a particular row is feasible is decided only after quantized angle
reconstruction passes both 20 mHa limits.

The numerical objective needs more care than simply minimizing lambda. The
verifier computes

`lambda_eff^2 = E_gap * (2*Lambda - E_gap)`.

The public R26 point sits just above the low `E_gap` floor and has a large
Lambda. On that branch, lowering Lambda while retaining the fit reduces lambda.
The verifier also permits `E_gap > Lambda`; on that branch moving E_gap toward
`2*Lambda` reduces lambda as well. There is no standalone lambda floor in the
verifier. A search should optimize the exact verifier expression and enforce
`Lambda>49`, `E_gap>3.914898`, `0<E_gap<2*Lambda`, and both residual limits on
the quantized words at every accepted step. Assuming 19.192 is a universal
lambda lower bound would miss the high-gap branch.

## Recommended implementation sequence

1. **Create the reliable ~9.0 notebook.** Keep our Q84 inverse square root and
   use the attached public Q932 DQI plus R26 chemistry. This establishes the
   stronger base and catches dataset/integration issues.
2. **Build a cheap estimator before more full verifications.** Given
   `(R,B,C,brot,J)`, compute the official state-preparation cost, address width,
   loader schedule, total Q/T, and exact quantized factor metrics. Reject a
   candidate before circuit generation when it cannot beat the incumbent.
3. **Train chemistry for cost, not rank alone.** Sweep around address thresholds
   `76+R*B <= 2048, 1024, 512` and all QROAM block-size transitions. Use
   group-pruning followed by refitting, then jointly reduce Lambda/lambda,
   quantize angles, and refit weights against the decoded vectors. Search
   precision and J only after the numerical factorization passes.
4. **Implement the DQI hybrid.** First port the attached block-shear extraction
   model into an executable circuit and validate it. Then profile its field
   products and replace the dominant reciprocal/conversion maps with the Gu
   tower arithmetic and phase scan. Target Q<=950 and T<=800k before spending
   time on smaller improvements.
5. **Use the failure budget intentionally.** Search fixed coefficient windows
   and gap-repair depth with millions of cheap classical samples; reserve the
   official 102,400-shot verifier for finalists. Track value, phase, and cleanup
   failures separately.
6. **Return to invsqrt last.** Its current factor is already the strongest of
   the public pair. Improvements are still valuable, but they cannot close the
   11.15x product gap by themselves.

## Implementation status

[`high_scoring_quantum_circuit_compilation_v4.ipynb`](high_scoring_quantum_circuit_compilation_v4.ipynb)
implements steps 1 and 2 as the submission path. It builds Q84/Q932/R26 from
embedded source, runs all official verifiers at full shot counts, and packages
only after the extra metric-format checks pass. The local Q932 build reproduced
932 qubits and 1,424,536 static Toffolis; a fresh 102,400-shot official run had
zero value and phase failures and a task factor of 16.58075.
The Q84 circuit also passed all 200,540,928 official shots at cost 155,756.409,
for a factor of 13.33348. Combining those fresh results with the public verified
R26 cost projects **8.99281** for that seed.

The estimator now reconstructs omitted R26 vectors, evaluates the exact
quantized factor metrics before circuit generation, reproduces Q=656 and
T=31,860, exposes the 512/1024/2048 address cliffs, and can exit early when a
candidate does not beat a supplied incumbent cost.

The notebook embeds the reviewed block-shear and J-fraction models for the next
DQI iteration. The block-shear model projects 1,262,247 Toffolis around 936
qubits, which remains above the 800k target and is not substituted for the
verified decoder. Chemistry pruning/refitting also remains outside the default
submission because neither the full FCIDUMP nor a lower-rank trained checkpoint
is embedded in this repository. These projected paths stay separate from the
verified submission until they produce circuits and factors that pass the
authoritative verifiers.

## Sources reviewed

- Babbush et al., [*Leveraging verifiable structure in quantum circuits to
  find more efficient compilations with agentic AI*](https://arxiv.org/abs/2610.23832),
  especially sections III.B-III.C and equations (43), (61)-(63).
- The competition's authoritative `verifiers/` code and
  `verifiers/dfthcsos/SCORING_README.md`, including the exact QROAM model and
  acceptance thresholds.
- The attached `quantum-circuits-8-95-lb-cost-breakdown.ipynb`, including its
  complete J-fraction decoder, four-address chemistry loader, official reports,
  and R26 provenance.
- Gu, Kemeny, Yelin, and Lukin,
  [*Discovering state-of-the-art quantum circuits with large language models*](https://arxiv.org/abs/2610.22367)
  and the public [andigu/uncompute](https://github.com/andigu/uncompute)
  implementation, especially the Reed-Solomon decoder's cost profile.
- Khattar et al., [*Verifiable quantum advantage via optimized DQI
  circuits*](https://arxiv.org/abs/2510.10967), the baseline decoder cited by
  the competition paper and Gu et al.
- Low et al., [*Fast quantum simulation of electronic structure by spectrum
  amplification*](https://doi.org/10.1103/PhysRevX.15.041016), which supplies
  the block-encoding context behind the chemistry objective.

The exact 17-20 solutions are private, so their implementations cannot be
audited. The conclusions above separate verified public measurements from
resource projections and follow directly from the published scoring equations.

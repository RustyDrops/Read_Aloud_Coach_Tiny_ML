# Read-Aloud Coach — Views + Specialists plan (v7 line)

Written 2026-11-19, modeled on Write It Right's current approach (multiple
views of the same data + small specialist double-checkers for named failure
cases), with the v7 Path A lesson applied: **no specialist is built until its
oracle headroom is measured on validation/dev data.** Phase 0 (below) already
ran that measurement for the first three specialists. Local CPU diagnostics;
no candidate trained yet. Nothing here is a keeper verdict — gates for the
next version are suggested at the end and must be frozen in PROVENANCE
before any training.

## Where v2 stands (measured, not remembered)

- Keeper v2: adult argmax 0.9088 (int8), verifier rule τ=0.30 / margin 0,
  62 KB int8, 42 classes (40 words + _other_ + _silence_).
- Training corpora (license-logged, the ONLY training sources for anything
  in this plan): Google Speech Commands v0.02 (CC BY 4.0) + LibriSpeech
  words mined by forced alignment from dev-clean / train-clean-100
  (CC BY 4.0). Deterministic code augmentation (incl. VTLP) of these is
  allowed. **Child corpora (JIBO, ACAD) are EVAL ONLY — never trained on.**
- Child speech (JIBO dev, 398 clips / 53 children): argmax **0.4497**,
  true-accept **0.3894**. Failure mode is false REJECTION (243 of 398 clips
  rejected); false-accept is only 0.0054 — the safety side is not the problem.
- The C4/G1 child EXAM is **SPENT** (v2 0.4427 / v6 0.4844 argmax). It must
  never be used as a gate again. Exam strategy is its own section below.
- v6's lever (VTLP augmentation of the whole model) gained +4.17 argmax
  against a +5 gate — real but insufficient. **No same-lever VTLP retry.**
  VTLP still appears below, but only as training data *for specialists*
  and as one view — a different, narrower use.
- 'a' scores 0.000 on child dev AND the spent child exam, for both v2 and
  v6. Adult 'a' is also broken: 0.1429 on v2's adult test — on n=7, because
  'a' is a LibriSpeech-mined schwa with almost no training clips. This is a
  data-starvation failure, not a child-pitch failure (Phase 0c).

---

## Part 1 — Views of one clip

Six views, analogous to the stroke model's six views. Each view gets a
**solo test on validation data only** before any fusion: train a tiny probe
(same small architecture family as v2, or logistic on view features) on
that view alone, from the v2 training corpora, and score it on (i) adult
validation and (ii) child DEV as a diagnostic. A view joins a fusion only
if its solo score shows signal the mel view lacks — specifically, if its
errors are *decorrelated* from the mel model's on child dev (it gets right
a meaningful share of clips mel gets wrong), not merely if it is accurate.

| # | View | What it sees that mel alone may miss |
|---|---|---|
| 1 | **Mel-spectrogram** (current) | The baseline view. Everything else is judged against it. |
| 2 | **Pitch / F0 contour** | Child-vs-adult register and intonation shape, normalized per clip. Sees *who is speaking* (a 5-year-old at ~260 Hz) independent of *what* — the register signal a rescue specialist needs. Phase 0c: F0 medians are ~250–270 Hz for every dev word, so pitch identifies the speaker class, not the word. |
| 3 | **Child-normalized mel** (VTLP-warped *input*, α≈1.2–1.3 applied at inference to map child speech toward the adult register v2 was trained on) | The inverse of v6's training augmentation: instead of teaching the model children, translate the child clip into the model's native register. Cheap (deterministic DSP), testable solo with zero training. |
| 4 | **Duration + energy envelope** | Word length and syllable rhythm at 10 ms resolution: 'a' is ~0.1–0.2 s of schwa; 'three' is long with a fricative head; 'no'/'go' are diphthong glides. Sees timing the 1 s mel window smears. |
| 5 | **Phonetic-contrast (vowel) view** | Formant-track features (F1/F2 trajectories from LPC or mel-band ratios) aimed at vowel identity — the dimension that separates 'a' (schwa) from 'on' (back vowel) and 'three' from 'tree' (the /θr/ onset + vowel length). Phase 0c caveat: as a *standalone* 'a' detector it did NOT beat mel (below) — it must earn its place in the solo test like every other view. |
| 6 | **Raw-waveform statistics** | Zero-crossing rate, spectral centroid/bandwidth, voiced fraction, crest factor — a compact "texture" summary, robust to the exact failure where mel features saturate on short clips. |

Solo-test protocol (frozen for all views): same train/val speaker split as
the v2 recipe; report solo argmax on adult val, solo argmax on child dev
(diagnostic only), and the decorrelation count (dev clips the view gets
right while v2's argmax is wrong). **Fusion rule:** at most the mel view +
the two best decorrelated views; a fusion that does not beat mel-only on
adult val by a pre-set margin is abandoned on paper, not tuned.

## Part 2 — Specialists (small double-checkers)

All specialists are target-conditioned where the product allows it:
Read-Aloud is a *verifier* — the expected word is known at inference — so a
specialist may use the target word as an input. That is a stronger design
than argmax re-ranking, and Phase 0 shows it is the necessary one for 'a'.

### S1 — Child-register second-chance (rescue) verifier
- **Named failure:** v2 falsely rejects child clips — 243/398 on dev;
  50 pushed to _other_, 168 to a wrong word, 24 right-but-quiet (argmax
  correct, p < 0.30).
- **Trigger:** v2's verifier rejects the target word AND the pitch view
  (View 2) classifies the clip as child-register (median F0 ≥ ~220 Hz or a
  learned register probe). Adult clips never trigger it, protecting the
  adult gate and the false-accept budget.
- **Job:** binary — "is this clip the target word, spoken by a child?"
  Trained ONLY on v2 corpora + VTLP-warped copies of them (child-like
  adults), never on child corpora. Warped-adult evidence it can work:
  on the α=1.25 warped-adult probe, v2 scores 0.826 and 61.5% of its
  false rejections have the true word at rank ≤ 2 — the same rescue
  geometry as real children.
- **Size budget:** ≤ 30 KB int8 (small CNN or logistic on view features).

### S2 — three/tree pair verifier
- **Named failure:** three→tree was 25.0% of 'three' clips under v2
  (16.1% under v6). On dev: 13 of 64 'three' clips argmax to 'tree', and
  the top-2 is exactly {three, tree} in 26 of 64.
- **Trigger:** target is 'three' or 'tree' AND v2's top-2 is exactly
  {three, tree} (either order).
- **Job:** binary three-vs-tree on the phonetic-contrast + duration views
  (the /θ/ onset and vowel length are exactly what mel smears). Trained on
  v2 corpora (Speech Commands has both words natively) + VTLP copies.
- **Size budget:** ≤ 20 KB int8.
- **Validation caveat (honest):** dev contains 64 'three' clips but only
  **1 'tree' clip** — headroom is measured one-directionally. The
  specialist's tree-side behaviour can only be gated on adult val +
  warped adults until a new child exam exists.

### S3 — 'a' verifier (REDESIGNED by Phase 0 — the pair-verifier design is dead)
- **Original idea (killed on paper):** an a/on pair verifier on v2's top-2.
  Phase 0b: in 49 dev 'a' clips, 'a' is in v2's top-2 **zero times**
  (top-3 also zero; median p('a') = 0.0042). A top-2 trigger would never
  fire. Oracle gain of that design: **0 clips.**
- **Redesign:** a target-conditioned binary verifier — consulted whenever
  the *target word is 'a'* and v2 rejects it. Separately, the root-cause
  data lever: mine 'a' at scale from LibriSpeech train-clean-100/360 with
  the existing forced-alignment pipeline ('a' is the most frequent word
  in English; the v2 recipe simply never collected more than a handful).
  The verifier trains on that mined adult 'a' + VTLP copies + hard
  negatives ('on', 'bird', 'go' — the measured confusion row, Phase 0c).
- **Size budget:** ≤ 20 KB int8.

## Part 3 — Headroom rule (mandatory, from Write It Right v7 Path A)

A specialist is built ONLY if its **oracle headroom** on child dev +
adult validation clears a stated margin, measured before any training:

- **Rescue-type specialists:** perfect-rescue oracle must promise
  ≥ **+5 pts true-accept** on child dev (the size of the gate it would
  need to move), counting only clips its trigger can actually reach.
- **Pair/target verifiers:** oracle must promise ≥ **+1.5 pts overall
  argmax** on child dev, or ≥ 10% of the specialist's named failure class,
  whichever is larger — and the trigger must demonstrably fire on dev.
- If the oracle fails the margin, the specialist is rejected ON PAPER
  (as v7's cascade was, and as S3's original design was below). No
  "build it and see."

## Part 4 — Phase 0 results (run 2026-11-19, local CPU, 0 units)

Method: v2 int8 re-scored on the JIBO **dev** set with full probability
capture (`child_eval/phase0_diagnostics.py` → `phase0_results.json`,
`phase0_dev_probs.npz`). Sanity: argmax 0.4497 and true-accept 0.3894
reproduce C4's recorded dev numbers **exactly**; on the warped-adult
probe the same pipeline reproduces G1's recorded v2 = 0.826 exactly.
The spent child exam was not touched. (One cross-check was discarded:
v2 on v3's adult test cache read 0.68, implausible against v2's recorded
0.91 — a mel-normalisation mismatch between cache generations. Adult
headroom below rests on metrics_v2's val record + the warped probe.)

### (a) False-rejection rescue headroom — **S1 PASSES, strongest specialist**
- 243 false rejections of 398 dev clips. Rank of the true word among them:
  rank 1 (right but quiet) 24 · rank 2 **48** · rank 3 23 · rank 4 16 ·
  rank 5 13 · **rank > 5: 119 (49% — no top-2/3 rescue can reach these).**
- Oracle true-accept: rescue rank ≤ 2 → **0.5704 (+18.1 pts)**;
  rank ≤ 3 → 0.6281 (+23.9 pts); quiet-only rescue → 0.4497 (+6.0 pts).
  Even a one-third-efficient S1 clears the +5 margin.
- Oracle argmax (perfect rank-2 promotion): +48 clips → 0.5704 (+12.1 pts).
- By word (FR with true rank ≤ 2): no 27, three 13, one 11, two 10,
  yes 2, **a 0, and 0** — S1 does not fix 'a' or 'and'; they need S3/data.

### (b) Pair-verifier headroom
- **three/tree — S2 PASSES.** 64 'three' clips: 13 argmax to 'tree',
  top-2 exactly {three, tree} in 26, true rank ≤ 2 in 41. Oracle pair
  gain: 8 argmax clips = **+2.0 pts overall** (12.5% of the 'three'
  class) — clears the +1.5 pt margin. Tree side unmeasurable on dev (n=1).
- **a/on — FAILS as designed.** 49 'a' clips: 13 argmax to 'on', but 'a'
  itself is never in the top-2 (or top-3); top-2 is exactly {a, on} in
  **0** clips. Oracle gain **0**. Dev also contains **zero** true-'on'
  clips, so the pair is not even validatable two-directionally on dev.
  → S3 redesigned as a target-conditioned verifier + data lever (above).

### (c) 'a' diagnosis — **data starvation, not a view problem**
- Full confusion row for the 49 child 'a' clips (v2 argmax): on 13,
  bird 11, go 7, tree 3, two 3, bed 3, house 2, little 2, cat 1,
  _other_ 1, is 1, the 1, happy 1 — scattered across 13 classes, no
  single rival to aim a pair verifier at. Median p('a') = **0.0042**:
  the model has essentially no 'a' concept for child voices.
- F0 does not separate 'a': median F0 by word on dev is 246–271 Hz for
  every word ('a' 258 Hz). Pitch tells you a child is speaking, not
  which word.
- Probe (child-disjoint folds within dev, logistic, diagnostic only):
  separating 'a' clips from all others — pitch/formant/duration features
  AUC **0.852**, mel-mean features AUC **0.887**, combined 0.861.
  Two conclusions: (1) the phonetic-contrast view does **not** beat mel
  at finding 'a' — no view shortcut; (2) 'a' clips ARE separable from
  other words by simple features (AUC ~0.85–0.89 on unseen children), so
  the signal exists in the audio — v2 never learned it because its
  training set barely contains the word (adult test n=7, acc 0.1429).
  **The 'a' fix is mined data + a target-conditioned verifier, in that
  order.**

### Headroom verdicts
| Specialist | Oracle promise | Margin | Verdict |
|---|---|---|---|
| S1 rescue | +18.1 pts true-accept (rank ≤ 2) | +5 | **BUILD (first)** |
| S2 three/tree | +2.0 pts argmax; fires on 26/64 | +1.5 | **BUILD (second)** |
| S3 a/on pair (original) | 0 clips | +1.5 | **REJECTED on paper** |
| S3 'a' target verifier + mined-'a' data | probe AUC 0.85–0.89 (signal exists); oracle n/a until data mined | re-measure after data lever | **BUILD after data lever, re-run headroom first** |

## Part 5 — Exam strategy for the next version (honest ranking)

The child exam is the binding constraint; the adult exam is easy
(LibriSpeech test-clean speakers + Speech Commands held-out give a fresh
adult exam on demand). Ranked by honesty:

1. **ACAD (Arizona Child Acoustic Database)** — new children, isolated
   words covering bed, cat, on, dog, go, one, two, three, house, bird —
   including the 'on' and 'tree'-side coverage dev lacks. Consent already
   documented and cleared in C4. **Blocked only on Rusty's free TalkBank
   signup** and download into `child_eval/`. This is the only fully clean
   new child exam currently identified. If it arrives, freeze it as the
   v7 child exam before any v7 training.
2. **A new consent-cleared child corpus not yet used** — none is in hand
   today; a survey (CHILDES/TalkBank siblings, Zenodo sets) is a
   legitimate plan step but must clear the same consent documentation
   as C4 before ingest. Do not promise it exists.
3. **Re-partitioned JIBO exam (development-grade only)** — all 110 JIBO
   children are already split between C4 dev and the SPENT G1 exam, so
   any re-split reuses children the candidates' predecessors were gated
   on. A re-split (e.g. by word within child, or dev/exam swap) is
   **writer-overlapping and must be labelled a development exam, never a
   keeper gate.** It can rank candidates; it cannot crown one.
4. **Warped-adult synthetic exam (supporting evidence only)** — unlimited
   and fresh (new α values / new adult speakers), and its rescue geometry
   matched real children in Phase 0 — but it is synthetic. Report it
   beside, never instead of, a real child exam.

**Consequence stated plainly:** until option 1 or 2 lands, v7 can be
*developed* and *ranked*, but a keeper verdict on children would be
dishonest. The plan therefore sequences the build so everything except
the final gate is ready when a clean exam exists.

## Part 6 — Suggested gates for the next version (to freeze in PROVENANCE)

- Child (NEW exam, option 1 or 2): argmax ≥ v2-on-that-exam + 5 pts AND
  true-accept ≥ v2 + 5 pts, both scored same-sitting.
- Adult: non-regression ≤ 1 pt vs v2 on a fresh adult exam; false-accept
  (all pairs) ≤ v2's + 0.5 pt, with the S1 trigger audited to fire on
  ~0% of adult clips.
- 'a' named check: child-exam 'a' true-accept > 0 (it is 0.000 on both
  measured child sets today) — reported by name whatever the headline.
- Size: total deployed (v2 + specialists + routing) ≤ 0.5 MB.
- Any specialist whose contribution is negative in ablation is removed,
  not re-tuned against the exam.

## Part 7 — Sequence and costs

1. **Phase 0 — DONE** (this file): headroom measured, S3-original killed,
   S1/S2 cleared. Local CPU, 0 units.
2. **'a' data lever** (local CPU, 0 units): extend the LibriSpeech mining
   pipeline to collect 'a' (and secondarily 'and'/'the'-class function
   words) at scale from train-clean-100/360; count what was obtained;
   re-run the S3 headroom probe with the new counts before building S3.
3. **View solo tests** (local CPU probes, 0 units; optional T4 if a view
   needs a small CNN probe: ≤ 0.2 units): mel vs View 3 (child-normalized
   mel — zero-training, test first) vs View 2/4/5/6 probes, per the
   frozen solo protocol. Kill views that don't decorrelate.
4. **Build S1, then S2** (T4, est. 0.15–0.5 units each from run history:
   v2's full training was 0.14 units; v6's job 0.02 units of T4 time;
   per-job hard cap 4 units, day cap 12): specialists trained on v2
   corpora + VTLP copies only, streaming/chunked merges (v5 OOM lesson),
   every phase's outputs pulled locally immediately.
5. **Assemble v7 candidate** = v2 + surviving views/specialists, ablation
   per component on adult val + child dev (diagnostic) + warped-adult
   exam (supporting). No child-exam scoring until a Part-5 option-1/2
   exam is frozen.
6. **Gate** on the new child exam when it exists; verdict in the usual
   card format.

**Stop rules:** a specialist that fails its ablation is dropped, not
re-tuned; no same-lever retries; if the mined-'a' count comes back tiny
(alignment failures), S3 stops and the 'a' problem is reported as a data
dead-end rather than modelled around.

## First implementable step

**The 'a' data lever (step 2) and the View-3 solo test can both start
immediately at zero cost** — but if only one: start with **S1's training
set build** (v2 corpora + VTLP copies, child-register trigger from the
pitch view), because S1 has the largest measured headroom (+18.1 pts
oracle true-accept), its trigger and training data already exist, and it
needs no new exam to develop — only to crown.

---

## Phase 1 results (run 2026-11-19, local CPU, 0 Colab units)

Steps 2, 3 (View-3 solo test), and the S1/S2 builds from Part 7 were run.
All child numbers below are JIBO **dev** numbers (development, labelled) —
the spent C4/G1 exam was never touched. Adult numbers are on the rebuilt
`cacheV2_val.npz` (the original cache files had been lost; `merge_v2.py`
is deterministic and the rebuild reproduces the recorded v2 val metrics
under the int8 model: argmax 0.9127 vs recorded fp32 0.9176, true-accept
0.9114 vs 0.9159, FA 0.0016 vs 0.0015 — the same int8 gap as on test).
Metrics files: `metrics_view3.json`, `metrics_s1.json`, `metrics_s2.json`,
`metrics_mine_a.json`, `metrics_s3.json`, `metrics_s3b.json`,
`metrics_assembly_dev.json`.

### View 3 (child-normalized mel) — WORKS, zero training
Inverse VTLP on the dev input, frozen v2 int8 scored unchanged:
| α | argmax | true-accept |
|---|---|---|
| 1.00 (unwarped) | 0.4497 | 0.3894 |
| **0.87** | **0.5226 (+7.3)** | **0.4749 (+8.6)** |
| 0.80 | 0.5050 (+5.5) | 0.4573 (+6.8) |
| 0.74 | 0.5126 (+6.3) | 0.4648 (+7.5) |
The single largest child gain found so far costs no training at all —
it is an input transform. (α here is the inverse of v6's training warp:
instead of teaching the model children, translate the child toward the
model. Related lever, opposite direction, and it stacks with nothing
yet — see the assembly diagnostic below.)

### S1 rescue verifier — BUILT; real but far below oracle (+3.0 of +18.1)
Logistic verifier (132 features: v2 prob features + mel band/temporal
stats + target one-hot), 5.7 KB stored — well inside the 30 KB budget.
Trained on v6-train pairs (originals + VTLP copies, group-disjoint
holdout). Threshold frozen on the holdout trigger population at
pair-FA ≤ 0.02 → 0.425; holdout rescue rate there is only 20.5% despite
all-pairs AUC 0.981 — within the trigger population (the clips v2 finds
hardest) the verifier's evidence largely overlaps v2's own.
- Child dev, gated (F0 ≥ 220 Hz; dev fire rate 0.877): true-accept
  0.3894 → **0.4196 (+3.0 pts)**, corrected argmax 0.4497 → 0.4623,
  false-accept 0.0054 → **0.0064** (+0.10 pt — at the top of v2's
  recorded band, not above the 0.0064 v6-exam figure).
- Adult val, UNGATED worst case (val cache has no waveforms): TA
  0.9114 → 0.9230, FA 0.0016 → 0.0019. Adult non-regression holds with
  a wide margin even if the gate always fired.
- Warped-adult probe (ungated): TA 0.819 → 0.851 (+3.2) — same ~1/6
  oracle efficiency as on real children, so the shortfall is in the
  verifier, not in dev.
- Gate audit (honest blemish): the raw F0 ≥ 220 Hz gate fires on
  **25.0%** of adult LibriSpeech val/test-speaker clips (adult women
  reach 220 Hz; F0 p50 167 Hz, p95 430 Hz incl. estimator octave
  errors). As built, the system is protected by the verifier's own
  strictness, not by the gate alone. A production v7 needs the plan's
  alternative — a learned register probe — or a higher/adaptive gate;
  the measured adult FA impact is small either way (above).

### S2 three/tree verifier — BUILT; small, clean, positive
Logistic on mel/duration/prob features (2.8 KB; budget 20 KB), holdout
AUC 0.966 / acc 0.90 on adults.
- Child dev: 26 triggered 'three' clips — corrected 2, broke 0
  (triggered-subset argmax 0.692 → 0.769; +0.5 pt overall dev argmax).
  Oracle was 8 clips; realized 2. Tree side on dev: n=1, unmeasurable,
  as pre-registered.
- Adult val (both directions measurable): 419 triggered — corrected 11,
  broke 1 (0.893 → 0.917 on the triggered subset). Net positive.

### 'a' data lever — data WON (960 clips), verifier transfer LOST
Mining (project alignment pipeline, train-clean-100, min-span fix
documented in `mine_a_tc100.py`): **960 new 'a' clips from 221 speakers**
(final), vs 80 in the old pool; 'a' inventory is now 1,040 vs the 64
originals v2 trained on. The scarcity was the 0.04 s minimum-span filter
discarding 1-2-frame schwas — a pipeline bug, not a corpus shortage.
S3 headroom re-measurement (verifier trained on 588 positives incl.
pool, threshold frozen on adult holdout, then scored once on dev):
- Probe 1 (negatives = SC isolated words, per the original plan):
  adult holdout AUC 1.000 — an artifact (short padded segments vs full
  1 s words) — and dev AUC 0.49, TA('a') 0.000. Worthless as built;
  reported, not shipped.
- Probe 2 (matched negatives: other LibriSpeech-mined words, same
  extraction pipeline, + SC on/bird/go as aux): adult holdout AUC
  0.933 (realistic), frozen threshold 0.925 — dev AUC **0.5746**,
  TA('a') still **0.000**, FA('a') 0.000. The verifier's probabilities
  on child clips saturate at ~0.000 (median and p90 both 0.0000):
  adult-trained features simply do not reach child 'a', VTLP copies
  notwithstanding. Phase 0's AUC 0.85-0.89 was *within-child*
  separability; adult→child transfer for this word is near chance.
- **Revised S3 verdict: the bolt-on 'a' verifier is a dead end on this
  evidence — do not build it.** The mined data is still the fix's raw
  material, but it has to enter through the MAIN model (a future
  whole-model retrain where 'a' is no longer starved, judged on a fresh
  child exam), not through a target-conditioned sidecar trained on
  adults.

### Assembly diagnostic (exploratory, NOT a gate)
View 3 (α=0.87) → v2 → S1 → S2 on dev: true-accept 0.3894 → **0.4849**,
argmax 0.4497 → **0.5302** (S1-corrected), FA 0.0054 → 0.0059. S1/S2
were trained on unwarped-input probs, so this is distribution-shifted
and optimistic/pessimistic in unknown places — it shows the pieces
compose, nothing more. (Dev mel was cached as float16 for this run;
argmax on warped input reads 0.5176 here vs 0.5226 in the float32
View-3 run — a cache-precision difference, disclosed.)

### What a v7 still needs (unchanged, and now sharper)
1. **A fresh child exam** (Part 5: ACAD via Rusty's TalkBank signup, or
   a new consent-cleared corpus). Everything above is development-grade
   until then; no keeper verdict on children is possible.
2. A whole-model retrain is the only remaining route for 'a' — with
   the 960 mined clips in the training mix (and the min-span fix kept
   for any future mining). Whether VTLP joins that retrain is a new
   pre-registration question; the no-same-lever rule bars re-rolling
   v6 as-is, and View 3 now offers an alternative way to spend the
   warp idea (at inference, for free).
3. A learned register probe if S1's gate is ever productised (the raw
   F0 gate leaks ~25% on adult women; system FA stayed safe only
   because the verifier is strict).

# Read-Aloud Coach — Tiny ML

![Read-Aloud Coach — read a word out loud, it checks the reading](promo/read-aloud-promo.png)

A tiny known-page word verifier for early readers (62 KB, runs fully
offline): the child reads a word from a known page aloud, and the model
says whether that word was read correctly — **READ** vs **STUMBLED /
NOT THIS WORD**. One small ONNX model, no network, no account. Trained
only on public, license-verified data — no AI-generated labels, no
frontier-model output in the training data. See PROVENANCE.md.

> **Validation status: validated on adult/teen voices, NOT yet on
> children 5–8.** Every training and test voice is an adult (LibriSpeech)
> or adult/teen (Google Speech Commands). Accuracy on the actual target
> users — children aged 5–8 — is unproven. Treat this as a supervised
> prototype: verdicts are hints for a supervising adult, not grades.

## How the verifier works

- Input: a 1-second 16 kHz clip → 40-band log-mel spectrogram (40×63).
- Model: depthwise-separable CNN (42,586 parameters), 42 classes = the 40
  covered vocabulary words + `_other_` (any other word) + `_silence_`.
- Verifier (`predict.py`): the app supplies the **target word** from the
  page being read. The word is accepted — verdict READ — only if it
  passes a decision rule tuned on the validation split (never on test):

      accept word e  iff  p(e) ≥ 0.30  and  p(e) − max_{c ≠ e} p(c) ≥ 0.0

  If the model's top prediction is a different covered word, the verdict
  is NOT THIS WORD (with the word it heard); anything else is STUMBLED.
  The rule was chosen to maximise true-accepts subject to a ≤10%
  false-accept budget on validation — it landed far inside the budget
  (see table), at the cost of rejecting about 1 in 10 correctly read
  words.

## Results (held-out test, speaker-disjoint, evaluated once)

| Metric | v1 | v2 |
|---|---|---|
| Overall accuracy | 82.5% (6,012 clips) | **91.1%** (6,413 clips) |
| Vocabulary-word accuracy | 82.7% | **91.0%** |
| — Speech Commands clips | 83.0% | **91.8%** |
| — LibriSpeech clips | 65.1% | **83.7%** |
| False-accept, plain argmax (v1 definition) | 18.4% | **8.1%** |
| False-accept with the decision rule | — (no rule) | **0.18%** |
| True-accept with the decision rule | — | **90.2%** |
| Words covered (of a 50-word list) | 26 | **40** |
| Model size (int8 ONNX) | 60 KB | **62 KB** |

False-accept = a clip that is *not* the target word being accepted as
read correctly — the failure that matters most for a reading coach,
because it tells a child a wrong reading was right. On the plain
argmax definition v1 used (a non-word clip classified as some
vocabulary word), v2 cuts it from 18.4% to 8.1%; with the decision
rule applied, only 0.18% of `_other_`/`_silence_` clips (0.15% over
all wrong-word pairs) are falsely accepted. The int8 model scores
essentially the same as fp32 (91.1% → 90.9% overall). v2 was trained
on a Colab T4 GPU (16 epochs, ~72 s of GPU time, 0.14 compute units)
from frozen speaker-disjoint caches: 37,769 train / 6,046 validation /
6,413 test clips. Full per-word numbers: `metrics_v2.json` (v1:
`metrics.json`).

## Phase 1 views + specialists — DEVELOPMENT (child DEV numbers only, not a release)

> **Development update, 2026-11-19. v2 (above) remains the keeper and
> the only released model in this repo.** Everything in this section is
> development work measured on the JIBO Kids **DEV** split only (398
> clips / 53 children, disjoint by child from the spent C4/G1 exam).
> These are **not** release numbers, there is no v7, and no new keeper
> is claimed. Full record: `PLAN-VIEWS-SPECIALISTS.md` (Phase 0 +
> Phase 1) and `PROVENANCE.md`.

Phase 1 tested the Write It Right pattern — several *views* of the
same clip plus tiny *specialist* double-checkers — against v2's
measured child-speech failure (dev argmax 0.4497, verifier
true-accept 0.3894, false-accept 0.0054; the failure mode is false
*rejection* of child clips, not false accepts).

- **View 3 — child-normalized mel (the big win, zero training).**
  Warp the *input* toward the adult register v2 was trained on
  (inverse VTLP, alpha = 0.87) and score the **frozen v2 model,
  unchanged** (`view3_normalized.py`, `metrics_view3.json`):
  child dev argmax **0.4497 -> 0.5226**, true-accept
  **0.3894 -> 0.4749**. The largest child gain found so far is an
  input transform, not a trained model.
- **S1 — child-register rescue verifier** (`train_specialists.py`,
  `s1_model.npz`, 5.7 KB; `metrics_s1.json`). A small logistic
  verifier consulted when v2 rejects a clip the pitch view flags as
  child-register. Child dev, gated: true-accept **+3.0 pts**
  (0.3894 -> 0.4196) — real, but only ~1/6 of its +18.1 pt oracle
  headroom, because its evidence overlaps v2's own. Cost:
  false-accept **0.0054 -> 0.0064** (top edge of v2's recorded band).
  Disclosed blemish: the raw F0 >= 220 Hz gate fires on **25.0%** of
  adult LibriSpeech clips (adult women reach 220 Hz); the system
  stayed safe because the verifier itself is strict (adult-val
  false-accept moved only 0.0016 -> 0.0019 even ungated). A product
  version would need a learned register probe, not this raw gate.
- **S2 — three/tree pair verifier** (`s2_model.npz`, 2.8 KB;
  `metrics_s2.json`). Consulted when v2's top-2 is exactly
  {three, tree}. Child dev: 26 triggered 'three' clips, corrected 2,
  broke 0. Adult validation (both directions measurable): 419
  triggered, corrected 11, broke 1 (0.893 -> 0.917 on the triggered
  subset). Small, clean, positive — the tree side cannot be measured
  on child dev (n=1), as recorded in the plan.
- **Exploratory assembly (diagnostic only, NOT a candidate).**
  View 3 warp -> v2 -> S1 -> S2 on dev: true-accept **0.4849**,
  argmax **0.5302**, false-accept **0.0059**
  (`metrics_assembly_dev.json`). S1/S2 were trained on
  unwarped-input probabilities, so this is distribution-shifted and
  shows only that the pieces compose.
- **The 'a' problem — root cause found, bolt-on fix rejected.**
  'a' scored 0.000 on every measured child set. The cause was a
  pipeline bug, not the corpus: the mining pipeline's **0.04 s
  minimum-span filter** discarded the word wholesale (a schwa aligns
  to 1-2 CTC frames). With the filter fixed, mining LibriSpeech
  train-clean-100 produced **960 new 'a' clips from 221 speakers**
  (`mine_a_tc100.py`, `metrics_mine_a.json`; total 'a' inventory
  1,040 vs the 64 originals v2 trained on). But a bolt-on
  target-conditioned 'a' verifier trained on adults does **not**
  transfer to children (dev AUC 0.5746, child 'a' true-accept still
  0.000; `metrics_s3.json`, `metrics_s3b.json`) — that design is a
  dead end. The mined data must enter through a future whole-model
  retrain instead.

**Standing statement:** a v7 cannot be crowned — on children, at
all — without a **fresh child exam**. The C4/G1 child exam is spent;
the only fully clean new exam currently identified is the Arizona
Child Acoustic Database, which is consent-cleared but blocked on a
free TalkBank signup to download. Until a clean exam exists, this
line can be developed and ranked, but not honestly released as a
child-validated improvement. **v2 remains the keeper.**

Phase 1 files: `view3_normalized.py`, `train_specialists.py`,
`assembly_dev.py`, `mine_a_tc100.py`, `s1_model.npz`,
`s2_model.npz`, `metrics_view3.json`, `metrics_s1.json`,
`metrics_s2.json`, `metrics_s3.json`, `metrics_s3b.json`,
`metrics_mine_a.json`, `metrics_assembly_dev.json`,
`child_eval/phase0_diagnostics.py`, `child_eval/phase0_results.json`.
No child audio, child manifests, or corpus data of any kind is
included in this repo — child corpora are eval-only, local, and
never redistributed.

## Vocabulary

Target: a 50-word early-reader list (`vocab.py`). **Covered: 40 of 50.**

- From Google Speech Commands (isolated-word recordings), 16 words:
  bed, bird, cat, dog, down, go, happy, house, no, on, one, stop,
  three, tree, two, yes.
- From LibriSpeech (words cut from read-aloud sentences by forced
  alignment against the corpus's own human transcripts), 24 words:
  a, and, away, can, come, for, here, in, is, it, little, look, make,
  me, my, not, said, see, the, to, up, we, where, you.
- Not covered (fewer than the 40-clip minimum in the public corpora;
  dropped, never padded): big, blue, find, funny, help, jump, play,
  red, run, yellow.

## Honest limits

- **Not validated on children.** All voices are adult/teen. Training
  includes programmatic pitch-up augmentation toward child-like pitch,
  but that is not a substitute for real child speech, and no
  permissively licensed children's speech corpus exists publicly.
  Do not present these accuracy figures as child accuracy.
- **Supervised prototype only.** Even with the decision rule, ~9.8% of
  correctly read words are rejected (the child is told to try again
  when they were right), and a small share of wrong readings pass.
  An adult should supervise; the model must not grade.
- **Some per-word scores are nearly meaningless.** 13 covered words
  have fewer than 20 test clips. The weakest — a (14%, n=7), we (26%,
  n=19), can (33%, n=6), look (50%, n=4) — rest on a handful of clips
  and could move a lot with more data; several 100% scores sit on
  single-digit counts too.
- **Function words are intrinsically harder.** Words cut from
  continuous LibriSpeech sentences score 83.7% vs 91.8% for isolated
  Speech Commands words — short words like "a", "the", "to" blur into
  their neighbours in real reading.
- Test speakers are disjoint from training speakers, but all speech
  comes from two corpora recorded in quiet conditions; real
  classrooms, tablets, and background noise will differ.

## Data & provenance

- **Google Speech Commands v0.02** — CC BY 4.0. Isolated words;
  official speaker-hash splits.
- **LibriSpeech dev-clean + train-clean-100** (OpenSLR) — CC BY 4.0.
  Clips cut by CTC forced alignment (open-weight wav2vec2 used as a
  locating tool only) against LibriSpeech's own human transcripts.

No Muse or OpenAI/Luna output was used in data, labels, selection,
or filtering at any stage; augmentation is purely programmatic. Full
record: `PROVENANCE.md`.

## Files

- `predict.py` — the v2 verifier demo (loads `model_v2_int8.onnx`;
  class list and mel stats are read from `metrics_v2.json`)
- `metrics_v2.json` / `metrics.json` — full held-out metrics for
  v2 / v1, including per-word accuracy and the decision rule
- `train_v2.py` — v2 training script; `train_v2_gpu.py` — the GPU
  variant actually run on Colab; `finalize_v2.py` — one-shot rule
  tuning (validation only), single test evaluation, ONNX export
- `prep_sc.py`, `prep_ls.py`, `prep_ls_v2.py`, `prep_tc100_v2.py`,
  `merge_v2.py` — data preparation (dataset downloads, forced
  alignment, cache building)
- `train.py`, `export_eval.py` — the v1 pipeline
- `vocab.py` — the 50-word target vocabulary
- `PROVENANCE.md` — data sources, licenses, and method
- The v2 int8 model — `model_v2_int8.onnx` (62 KB) — **is in this
  repo** (root directory, next to `predict.py`). The fp32
  `model_v2.onnx` (171 KB) is not included; it is available on
  request.

## Run it (Python)

```bash
python3 predict.py <clip.wav> <expected_word>
```

Verified against the v2 int8 model on held-out Speech Commands test
clips:

```
$ python3 predict.py cat_test_clip.wav cat
predicted: cat (1.00)
expected 'cat' probability: 1.00
verdict: READ

$ python3 predict.py cat_test_clip.wav dog
predicted: cat (1.00)
expected 'dog' probability: 0.00
verdict: NOT THIS WORD (heard 'cat')
```

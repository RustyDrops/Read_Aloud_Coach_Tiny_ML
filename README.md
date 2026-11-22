# Read-Aloud Coach — Tiny ML

![Read-Aloud Coach — read a word out loud, it checks the reading](promo/read-aloud-promo-v3.png)

A tiny known-page word verifier for early readers (62 KB, runs fully
offline): the child reads a word from a known page aloud, and the model
says whether that word was read correctly — **READ** vs **STUMBLED /
NOT THIS WORD**. One small ONNX model, no network, no account. Trained
only on public, license-verified data — no AI-generated labels, no
frontier-model output in the training data. See PROVENANCE.md.

> **Validation status: validated on adult/teen voices; v4 is also
> evaluated on real child speech.** v4's training adds clips from 53
> children (JIBO Kids Corpus, ages 4–7), and it was scored on a
> held-out exam of 53 *different* children from the same corpus:
> **82.8%** accuracy, **82.0%** true-accept with the decision rule
> (see "Child evaluation" below). That is one research corpus of
> children reading single words to a social robot — encouraging, and
> the first child evidence this project has — but it is **not** a
> clinical or educational validation, and accuracy on any given
> child will vary. Treat this as a supervised prototype: verdicts
> are hints for a supervising adult, not grades.

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

| Metric | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| Overall accuracy | 82.5% (6,012 clips) | 91.1% (6,413 clips) | 92.4% (6,413 clips) | **92.7%** (6,413 clips) |
| Vocabulary-word accuracy | 82.7% | 91.0% | 92.3% | **92.5%** |
| — Speech Commands clips | 83.0% | 91.8% | 93.3% | **93.5%** |
| — LibriSpeech clips | 65.1% | 83.7% | 82.3% | **83.7%** |
| False-accept, plain argmax (v1 definition) | 18.4% | 8.1% | 6.8% | **6.0%** |
| False-accept with the decision rule | — (no rule) | 0.18% | 0.14% | **0.11%** |
| True-accept with the decision rule | — | 90.2% | 91.7% | **91.8%** |
| Words covered (of a 50-word list) | 26 | 40 | 40 | **40** |
| Model size (int8 ONNX) | 60 KB | 62 KB | 62 KB | **62 KB** |

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
6,413 test clips. Full per-word numbers: `metrics_v3.json` (v2:
`metrics_v2.json`, v1: `metrics.json`).

**v3 (2026-11-21)** changes exactly one thing versus v2: the
training data. Every training clip is duplicated with a
register-shifted copy (pitch/tempo resampled x1.26), teaching the
model that the same word arrives in higher, faster voices. Same
architecture, same frozen decision rule (tau 0.30, margin 0.0, not
retuned), same size class. Gains concentrate in Speech Commands
(+1.5 pts) and in the decision rule's true-accept (+1.5 pts at
lower false-accept); LibriSpeech clips dip slightly (83.7% ->
82.3%). The validation status at the top of this README is
unchanged and applies to v3 in full: **validated on adult/teen
voices, NOT on children 5-8.** On the JIBO child DEV split (398
clips, development only, not a release metric) v3 reaches argmax
0.4598 / true-accept 0.4322 — the best child-dev true-accept of any
whole model tried — with the word 'a' still at 0.000 (see
PROVENANCE.md, 2026-11-21 campaign).

## Child evaluation (v4, 2026-11-21)

v4 is the first version trained with real child speech, and the
first with a held-out child evaluation. Training added 398 clips
from **53 children** (the project's former dev split, retired as an
evaluation when it became training data). The evaluation is a
frozen exam of **384 clips from 53 different children** (ages 4–7,
pre-K to grade 1), disjoint from every training child, scored once:

| Child exam (int8) | v4 | best prior model (no child training) | v2 |
|---|---|---|---|
| Overall accuracy | **82.8%** | 51.0% | 44.3% |
| True-accept with the decision rule | **82.0%** | 47.7% | 40.4% |
| False-accept (all word pairs) | **0.31%** | 0.51% | 0.64% |
| The word "a" (53 clips) | **90.6%** | 0.0% | — |

The "a" row is the story of this release: across sixteen prior
experiments the word scored exactly zero on child speech, because
children say a long open vowel where every adult training clip has
a 56 ms article blip. Only real child examples fixed it. Adult
accuracy did not suffer (table above: v4 is also the best adult
model in the repo).

**Attribution — JIBO Kids Corpus.** Child speech data: Shankar et
al. (2024), "The JIBO Kids Corpus," *JASA Express Letters* 4(11),
115201 (DOI 10.1121/10.0034195); corpus distribution via Zenodo
(DOI 10.5281/zenodo.13964792, v0.1). Word-level transcripts:
DrivenData "JIBO Kids Corpus Transcripts" (K-12 AI Infrastructure
Platform), CC BY 4.0. Corpus use terms per the DrivenData dataset
listing: commercial and noncommercial use with attribution.
Children in the corpus are anonymized; no child audio is
redistributed in this repo — only the trained model and aggregate
scores.

## How this compares

The standard tiny-model reference for this task family is keyword
spotting on Google Speech Commands:

- **MLPerf Tiny** (the industry TinyML benchmark suite) uses a
  DS-CNN with 38.6k parameters as its keyword-spotting reference,
  sets a **90% top-1 quality target**, and measured **92.2%** for
  that reference model. (Banbury et al., "MLPerf Tiny Benchmark,"
  NeurIPS Datasets & Benchmarks 2021 —
  https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/da4fb5c6e93e74d3df8527599fa62642-Paper-round1.pdf)
- The original **DS-CNN paper** (Zhang et al. 2017, "Hello Edge:
  Keyword Spotting on Microcontrollers," arXiv:1711.07128 —
  https://arxiv.org/pdf/1711.07128v2) reached **95.4%** on Speech
  Commands keyword spotting.

Read-Aloud Coach v2 scores **91.1%** — inside that tiny-model
range — but on a harder task than 12 isolated keywords: **40
target words**, mixing isolated-word recordings with words cut
from continuous read speech (LibriSpeech), scored with a
false-accept-tuned decision rule (rule false-accept 0.18% at
90.2% true-accept on the held-out test). Same size class as the
references: 42,586 parameters, 62 KB int8.

And the standing caveat, updated for v4 but not softened: those
benchmarks are adult-voice benchmarks. v4 now has real child
evidence behind it (see "Child evaluation" above) — one corpus,
ages 4–7, single words read to a robot. That is a genuine held-out
result, not a general guarantee for children 5–8 in classrooms and
kitchens, which remains unproven (see "Validation status" at the
top and "Honest limits" below).

## Phase 1 views + specialists — DEVELOPMENT (child DEV numbers only, not a release)

> **Development update, 2026-11-19. At the time of writing v2 was
> the keeper; v3 superseded it on 2026-11-21, and v4 (above)
> superseded v3 the same day with the first child-trained model.** Everything in this section is
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

## Use it yourself

> **Warning, repeated on purpose: validated on adult/teen voices,
> NOT yet on children 5–8.** Use it as a supervised prototype —
> verdicts are hints for a supervising adult, not grades — and
> expect child accuracy to differ from the numbers above.

You need three files from this repo, kept together (the script
looks for the other two next to itself, or one directory up):

- `predict.py` — the verifier
- `model_v2_int8.onnx` — the shipped model (62 KB, int8; this is
  the model `predict.py` loads). The fp32 `model_v2.onnx`
  (171 KB) is **not** in this repo; it is available on request
  and scores essentially the same (91.1% vs 90.9% overall).
- `metrics_v2.json` — supplies the 42-class list (the 40 covered
  words + `_other_` + `_silence_`) and the log-mel normalisation
  statistics the model expects.

Install the Python dependencies (Python 3):

```bash
pip install numpy soundfile torch torchaudio onnxruntime
```

Run the verifier with a 1-second (or shorter — it is trimmed /
padded) **16 kHz mono WAV** of one spoken word, plus the word
the page asked for:

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

### What the verdicts mean

The verifier applies the decision rule tuned on validation only
(τ = 0.30, margin = 0.0):

```
accept word e  iff  p(e) ≥ 0.30  and  p(e) − max_{c ≠ e} p(c) ≥ 0.0
```

- **READ** — the clip passed the rule for the expected word: the
  model is confident this word was read as written.
- **NOT THIS WORD (heard 'x')** — the model's top prediction is a
  *different* covered vocabulary word, named in the verdict.
- **STUMBLED** — the model could not confirm the expected word:
  the top prediction was `_other_` / `_silence_`, or the expected
  word led but below the τ = 0.30 confidence threshold. Treat it
  as "not confirmed — try again," not as proof of a misreading:
  with this rule about 1 in 10 correctly read words is rejected
  (true-accept 90.2% on the adult/teen test set).

To use the model without `predict.py`, load
`model_v2_int8.onnx` in any ONNX runtime: feed a 40×63 log-mel
spectrogram (16 kHz, n_fft 512, hop 256, 40 mel bands, log,
normalised with the `mel_mean` / `mel_std` in `metrics_v2.json`)
as input `mel`, and map the 42 logits through the class list in
the same file. Apply the τ = 0.30 rule above if you want the
verifier behaviour rather than a plain argmax.

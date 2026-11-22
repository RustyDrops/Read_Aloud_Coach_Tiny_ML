# Provenance — Read-Aloud Coach prototype

**Validation label: validated on adult/teen voices, not yet on children 5–8.**

## Training data (only the following sources)

1. **Google Speech Commands v0.02** — licence: Creative Commons
   Attribution 4.0 (CC BY 4.0), verified from the `LICENSE` file inside
   the downloaded archive (speech_commands_v0.02.tar.gz, from
   storage.googleapis.com/download.tensorflow.org). Adult/teen crowd
   recordings of single words. Covers 16 of the 50 target words directly:
   bed, bird, cat, dog, down, go, happy, house, no, on, one, stop, three,
   tree, two, yes. 18 non-target words from the same release form the
   `_other_` class; the dataset's `_background_noise_` recordings form
   the `_silence_` class. Splits: the dataset's official
   validation_list.txt / testing_list.txt (speaker-hash based, so test
   speakers are disjoint from train speakers).
2. **LibriSpeech dev-clean** (OpenSLR resource 12) — licence: CC BY 4.0
   (LibriSpeech is distributed under CC BY 4.0; audio from LibriVox
   volunteer readings of public-domain books). Adult read speech.
   Isolated clips for the remaining common target words were cut by CTC
   forced alignment against the corpus's own human transcripts (see
   "Tools" below). Speaker-disjoint split by LibriSpeech speaker id.
   500 dev-clean utterances (≤12 s each) were aligned, yielding 1,648
   clips. Words kept (≥40 clips): and 247, for 65, in 124, is 65, it 94,
   my 48, not 59, the 462, to 211, you 61. Words below the threshold were
   dropped rather than padded — closest misses: we 37, said 30, me 29.
   Because the kept LibriSpeech words have far fewer clips than the
   Speech Commands words, LibriSpeech training clips were oversampled
   ×6 (duplication of existing training clips only) to keep those
   classes from being ignored by the model.

Common Voice (CC0) was permitted but not used in this build.

## Augmentation (programmatic only, script-generated)

- Pitch-up copies via resampling (factors 1.2/1.3/1.4) — a crude
  approximation toward child-like pitch, applied to 60% of Speech
  Commands training words.
- Background-noise mixing at 5–20 dB SNR using the dataset's own
  background-noise recordings.
- SpecAugment-style frequency/time masking and time rolls during training.

## Tools (not teachers)

- **Wav2Vec2 base ASR (torchaudio WAV2VEC2_ASR_BASE_960H, open weights)**
  was used only to *locate* words in LibriSpeech audio by forced
  alignment. All labels come from LibriSpeech's human transcripts. No
  model output was used as a label, score, ranking, or filter anywhere.
- No Muse output and no OpenAI/Luna output appears in the data, labels,
  selection, or filtering of this project. (Luna/Muse involvement in
  the wider pilot is code, QA, and post-hoc evaluation only.)
- No synthetic child speech: no TTS voices were passed off as children.
  Adult TTS was not used at all in this build.

## Not done / not claimed

- Not evaluated on children aged 5–8 (no permissively licensed corpus
  exists — see ../AUDIT-speech.md). Do not present accuracy figures as
  child accuracy.
- Not published anywhere. Hugging Face upload requires Rusty's approval
  of the exact version.

---

# v2 (2026-11-18)

**Validation label (unchanged): validated on adult/teen voices, not yet
on children 5–8.**

## What changed vs v1

- **More data, same two sources, same licences.** LibriSpeech usage
  expanded from 500 dev-clean utterances to all 2,362 dev-clean
  utterances ≤12 s, plus a train-clean-100 top-up pass (3,000 of 7,727
  utterances scanned) aimed at near-miss words. Clips were cut by the
  same CTC forced-alignment method against the corpora's own human
  transcripts (wav2vec2 as a locating tool only — no model output used
  as label, score, ranking, or filter). An extractor bug found in v2
  prep (zero-frame-token spans silently dropping words with doubled
  letters, e.g. the EE in "see") was fixed in code; labels still come
  only from transcripts. Speech Commands side unchanged (clean local
  copy `data/sc_mine/` re-verified: 105,835 wavs, 0 bad).
- **Coverage: 40 of 50 target words** (v1: 26) — the 16 Speech Commands
  words plus 24 LibriSpeech words (a, and, away, can, come, for, here,
  in, is, it, little, look, make, me, my, not, said, see, the, to, up,
  we, where, you). Dropped under the same ≥40-clip rule, never padded:
  big, blue, find, funny, help, jump, play, red, run, yellow.
- **Frozen speaker-disjoint caches** (`cacheV2_*.npz`): train 37,769 /
  val 6,046 / test 6,413 clips, 42 classes (40 words + `_other_` +
  `_silence_`). LibriSpeech oversample reduced from ×6 (v1) to ×2.
- **Model:** same depthwise-separable CNN family, slightly wider
  (42,586 params, v1: 40,780). Same programmatic augmentation
  (SpecAugment-style masking, time rolls; pitch/noise augmentation is
  baked into the caches as in v1's pipeline).

## Training run (Colab GPU, 2026-11-18)

- The local CPU run was stopped after epoch 5 by Rusty's priority call
  and superseded: v2 was trained **fresh from epoch 0** on Google Colab
  (T4 GPU) using `train_v2.py`'s exact configuration (16 epochs, AdamW
  lr 2e-3, cosine T_max=16, label smoothing 0.05, batch 256) via
  `train_v2_gpu.py` — identical except device placement and checkpoints
  saved as CPU tensors. The frozen caches were transferred with sha256
  verification and clip counts re-checked on the runtime before
  training (37,769 / 6,046 / 6,413 — exact match).
- All 16 epochs completed in ~72 s of GPU time; best checkpoint =
  epoch 14, validation accuracy **0.9176** (v1 val best: 0.8295).
  Compute cost: **0.14 Colab units** (balance 597.69 → 597.55); runtime
  deleted and verified gone afterwards.
- One compatibility note: the runtime's torch 2.14 requires
  `dynamo=False` for the legacy ONNX export call in `finalize_v2.py`;
  this was verified on a dummy network before finalize ran, and only
  the runtime's copy of the export call was patched. Export semantics
  (opset 13, dynamic batch axis) unchanged.

## Final evaluation (finalize_v2.py — run exactly once, 2026-11-18)

Decision rule tuned on **validation only**: accept expected word iff
p(word) ≥ τ and p(word) − max-other ≥ margin, maximizing true-accept
subject to validation false-accept ≤ 0.10. Chosen: τ = 0.30, margin =
0.0 (validation FA 0.0015, true-accept 0.9159). The held-out test set
was then evaluated **once** (fp32 + int8 in the same pass):

- **Test overall accuracy 0.9113** (v1: 0.8253); vocabulary-word
  accuracy **0.9104** (v1: 0.8271). Per source: Speech Commands 0.9183
  (v1: 0.8299), LibriSpeech 0.8373 (v1: 0.6513).
- **False-accept:** plain argmax (v1's headline definition — non-word
  clips classified as a vocabulary word): **0.0813** (v1: 0.1837).
  With the tuned decision rule: 0.0018 on `_other_`/`_silence_` clips,
  0.0015 over all (clip, wrong-word) pairs, at a true-accept of 0.9015
  (false-reject 0.0985). The ≤10% false-accept budget is met on every
  definition. int8 ONNX: overall 0.9088, rule FA 0.0017, true-accept
  0.8977.
- Weakest words (small test counts, read with care): a (0.14, n=7),
  we (0.26, n=19), can (0.33, n=6), look (0.50, n=4).
- Artifacts: `model_v2.onnx` (170,938 B), `model_v2_int8.onnx`
  (62,151 B), `metrics_v2.json`, checkpoints in `ckpt_v2/`
  (best.pt = epoch 14). Full numbers in `metrics_v2.json`.

## Terms safety (v2)

Same rules as v1, unchanged: no Muse or OpenAI/Luna output in data,
labels, selection, or filtering; only the two CC BY 4.0 corpora above;
augmentation purely programmatic. Nothing published.

---

# v3 (2026-11-18) — PRE-REGISTRATION (written before v3 training and
# before the v3 test set was evaluated)

**Validation label (unchanged): validated on adult/teen voices, not yet
on children 5–8.**

**Priority target (Rusty):** v2's ~9% per-word error. At ~9% per word a
12-word page gets about one word wrong — too high. v3's job is to cut
per-word error while keeping false-accepts low.

## Test discipline

- v2's test cache (`cacheV2_test.npz`) is **SPENT**. It is never
  evaluated on, tuned against, or trained on in v3. No clip and no
  speaker from it is used anywhere in v3 (LibriSpeech pool speakers with
  md5%10 == 9 — the v2 test assignment — are excluded from v3 entirely).
- **Fresh v3 test set:** built from previously unused material only —
  LibriSpeech **test-clean** (OpenSLR 12, CC BY 4.0; its 40 speakers are
  disjoint from dev-clean / train-clean-100 speakers by LibriSpeech
  construction, and none appear in any v1/v2 cache). Clips are cut by
  the same CTC forced-alignment method against test-clean's own human
  transcripts (wav2vec2 as locating tool only). All 50 vocab words are
  extracted; at cache-build time, clips of words that are v3 classes
  keep their label, clips of non-vocab words become `_other_`, and
  clips of vocab words that are NOT v3 classes are excluded (not folded
  into `_other_`). `_silence_` clips are fresh slices (seed 777,
  offsets/scales disjoint in procedure from v1/v2's seed-11 slices) of
  the same Speech Commands `_background_noise_` recordings. Speech
  Commands testing_list clips are NOT used (v2's test cache already
  consumed them). The test cache and its manifest (per-word counts,
  speaker list, sha256) are frozen and recorded here before training
  starts; it is evaluated **exactly once**, after training, in the
  finalize pass.
- Validation for model selection and decision-rule tuning: the v2
  validation cache plus newly mined validation-split clips (same md5%10
  == 8 rule), clean only. The decision rule keeps v2's form (accept iff
  p(word) ≥ τ and margin over runner-up ≥ m), tuned on validation only,
  false-accept budget ≤ 0.10 on validation as in v2.

## Pre-registered gates (ALL must pass or v3 is rejected, v2 stays
## keeper, and there is no re-tuning against the fresh test)

1. Fresh-test **overall accuracy ≥ 0.93**
2. Fresh-test **vocabulary-word accuracy ≥ 0.93**
3. Fresh-test **false-accept ≤ 0.05** (v1 definition: `_other_` /
   `_silence_` clips whose argmax is a vocabulary word)
4. **Coverage ≥ 40 words** (the ≥40-clips-in-pool bar stands; recovering
   dropped words is welcome only if the pool honestly supports it)
5. **int8 ONNX ≤ 0.5 MB**

## Planned levers (deterministic code only; CC BY data only; no
## frontier-model output anywhere, as before)

- Finish the train-clean-100 scan (3,000 / 7,727 ≤12 s utterances were
  scanned for v2) plus a wanted-words pass over longer train-clean-100
  utterances, to deepen starved classes and try to recover dropped
  words (big, blue, find, funny, help, jump, play, red, run, yellow).
- Wave-level augmentation of real training clips: speed perturbation
  (0.9 / 1.1), background-noise mixing (5–20 dB SNR), synthetic
  exponential-decay reverb, in addition to v1/v2's pitch copies and
  in-training SpecAugment / time rolls.
- Modestly larger DS-CNN (≤ ~300 k params, int8 ≤ 0.5 MB) and more
  epochs (24–32) with the same optimizer family; checkpoint selection
  on validation accuracy only.
- Training compute: Google Colab GPU, cap 8 units for this round,
  started only when no other job holds a Colab assignment.

## Results (finalize_v3.py — run exactly once, 2026-11-18)

**Verdict: v3 REJECTED — v2 stays keeper.** 2 of 5 pre-registered
gates passed. No re-tuning against the fresh test was done or will be
done; the test is spent for this design.

Execution note: the run took three attempts (two drivers died; one
corpora pass was interrupted at 63 min). Attempt 3's runtime was
adopted after its driver died and the remaining phases were driven
manually in order — corpora → prep_pool → prep_test → merge → train →
finalize — with finalize run exactly once, as registered. Training:
28 epochs on T4, best checkpoint epoch 22, validation accuracy
**0.9453**. Compute: attempts 1–2 ≈ 1.40 units; attempt 3 ≈ 2.6 units
(shared-account window; the T4's own share ≈ 2.4–2.6) — **total
≈ 4.0 of the 8-unit cap**. All artifacts were pulled back and verified
locally (byte-size match + load checks) before the runtime was
deleted.

Decision rule tuned on **validation only**: τ = 0.30, margin = 0.0
(validation FA 0.0007, true-accept 0.9419). The fresh test set
(LibriSpeech test-clean, 7,887 clips, 40 speakers, sha256
a16aaf01…) was then evaluated **once** (fp32 + int8 in the same pass):

- **Test overall accuracy 0.7444** (gate ≥ 0.93 ✗); **vocabulary-word
  accuracy 0.8195** (gate ≥ 0.93 ✗). Per source: LibriSpeech
  test-clean 0.7348, silence 0.9867.
- **False-accept (v1 argmax definition): 0.7191** (gate ≤ 0.05 ✗).
  With the tuned decision rule the same test gives FA 0.0101 on
  `_other_`/`_silence_` clips (0.0041 over all pairs) at a true-accept
  of 0.8073 — but gate 3 was registered on the argmax definition and
  stands as failed.
- **Coverage 49 words ✓** (all nine recoverable dropped words are back;
  only `jump` has no class). **int8 ONNX 173,185 B ✓** (fp32 605,236 B;
  151,955 params). int8 test overall 0.7492.
- Diagnosis, from the per-word table: the fresh test is continuous
  read speech only, and Speech Commands vocabulary words cut from
  continuous speech collapse — `on` 0.004 (n=249), `one` 0.07,
  `two` 0.03, `no` 0.16, `go` 0.33 — the model learned those words as
  isolated utterances. The `_other_` class (800 real non-vocab words
  from test-clean) is open-vocab: a closed-vocab argmax assigns 72%
  of them to some vocabulary word. The v2-style validation mix
  (val 0.9453) did not predict either failure. Recovered LibriSpeech
  words did fine (`big`, `find`, `help` all 1.00 on small n).
- Lesson for any v4: the exam and the training distribution must both
  cover continuous-speech cuts of every class, and `_other_` needs
  far more diverse real-word coverage — a fresh test set would be
  required, since this one is spent.

Artifacts retained locally for the record: `metrics_v3.json`,
`model_v3.onnx`, `model_v3_int8.onnx`, `classes_v3.npy`,
`cacheV3_test.npz`, `ckpt_v3/` (best.pt = epoch 22), phase logs.
v2 remains the shipped model; nothing published.

## Terms safety (v3)

Same rules as v1/v2, unchanged: no Muse or OpenAI/Luna output in
data, labels, selection, or filtering; only CC BY 4.0 corpora
(Speech Commands, LibriSpeech incl. test-clean) with deterministic
code augmentation; wav2vec2 used as a locating tool only. Nothing
published.

---

# v4 (2026-11-18) — PRE-REGISTRATION (written before v4 training and
# before the v4 exam was built or evaluated)

**Validation label (unchanged): validated on adult/teen voices, not yet
on children 5–8.**

**Design basis — v3's diagnosis.** v3's fresh exam (LibriSpeech
test-clean, continuous read speech only) exposed a training-distribution
flaw, not a capacity flaw: (1) the 16 Speech Commands words had **zero**
continuous-speech training clips — v3's pool mining targeted only
non-SC words, so those words were learned as isolated utterances and
collapsed when cut from continuous speech (`on` 0.004, `one` 0.07,
`two` 0.03); (2) training `_other_` was only Speech Commands' 18
isolated non-target words, while exam `_other_` was 800 real non-vocab
words cut from continuous speech — argmax false-accept hit 0.7191.
With a validation-tuned rule, v3's false-accept was 0.0101 at 0.8073
true-accept: the capability is partly there; the distribution failed.
v4 changes the data distribution and nothing else about the recipe.

## v4 design (all registered here, before any v4 run)

1. **Continuous speech first, for every word.** A new mining pass over
   LibriSpeech dev-clean + train-clean-100 (same utterance filters as
   v3's pool pass: ≤12 s, plus 12–30 s utterances containing a target
   word) extracts clips for **all 50 vocab words** by the same CTC
   forced-alignment method against the corpora's own human transcripts
   (wav2vec2 as locating tool only). Per-word extraction cap 800.
   Speech Commands clips become secondary: for each SC-covered word,
   SC training clips are capped at that word's LibriSpeech
   train-split base count, so SC never exceeds the continuous-speech
   count for any word, and SC clips get no merge-time augmentation
   variants.
2. **`_other_` much larger and more diverse.** The same mining pass
   also collects non-vocab words (alphabetic tokens of length ≥ 2 not
   in VOCAB_50) cut from continuous speech: cap 20 per distinct word,
   14,000 candidates total. Training `_other_`: train-split speakers
   only, base cap 8,000, each base clip plus one deterministic
   augmented variant. SC `_other_` is capped at 2,000 clips.
3. **Class rule (unchanged shape):** a word is a class iff it has ≥ 40
   clips in the v4 mining pool; words below the bar are dropped, never
   padded. Speaker split for pool material (unchanged): LibriSpeech
   speaker id md5%10 — train ≤ 7, == 8 unused for training, == 9
   excluded entirely (v2's test assignment).
4. **Fresh exam — LibriSpeech test-other** (OpenSLR 12, CC BY 4.0),
   never used by any prior version. Extraction mirrors v3's exam
   procedure (all-word CTC extraction, vocab cap 600/word, `_other_`
   candidates cap 30/distinct word and 1,200 total, sampled to 800
   with seed 99; `_silence_` = 300 fresh slices of the Speech Commands
   background-noise recordings, seed 2026 — disjoint in procedure from
   v1/v2's seed 11 and v3's seed 777). **Disjointness is verified from
   the extracted data before the exam is frozen:** exam speakers must
   be disjoint from (a) v3's spent test-clean speakers, (b) all v4
   training-pool speakers, (c) v4 validation speakers; any overlapping
   speaker's clips are dropped before freezing and the drop recorded
   in the exam manifest. The exam cache and its manifest (per-word
   counts, speaker list, sha256) are frozen and recorded before
   training starts; the exam is evaluated **exactly once**, in the
   finalize pass. v3's exam is spent and is never re-used; v3's
   training pool is training material only.
5. **Validation — LibriSpeech dev-other** (same extraction procedure,
   vocab cap 400/word, `_other_` sampled to 800 with seed 98,
   `_silence_` seed 2027), frozen before training. Checkpoint
   selection and decision-rule tuning use this validation set only —
   unlike v3, it matches the exam's source and conditions.
6. **Recipe otherwise unchanged from v3:** same DS-CNN architecture
   (~152 k params), 28 epochs, AdamW lr 2e-3 cosine, label smoothing
   0.05, batch 256; merge-time deterministic variants for LibriSpeech
   clips (speed 0.9/1.1, noise mix 5–20 dB SNR from the v3 noise bank,
   synthetic reverb for starved classes); in-training SpecAugment /
   time rolls. Decision rule keeps v2/v3's form (accept iff
   p(word) ≥ τ and margin over runner-up ≥ m), tuned on validation
   only with false-accept budget ≤ 0.10, and is reported alongside —
   the argmax gate below is the gate.

## Pre-registered gates (ALL must pass or v4 is rejected, v2 stays
## keeper, and there is no re-tuning against the fresh exam)

1. Fresh-exam **overall accuracy ≥ 0.93**
2. Fresh-exam **vocabulary-word accuracy ≥ 0.93**
3. Fresh-exam **false-accept ≤ 0.05** (v1 definition: `_other_` /
   `_silence_` clips whose argmax is a vocabulary word)
4. **Coverage ≥ 40 words** (aim: keep v3's 49)
5. **int8 ONNX ≤ 0.5 MB**

## Terms safety (v4)

Same rules as v1–v3, unchanged: no Muse or OpenAI/Luna output in
data, labels, selection, or filtering; only CC BY 4.0 corpora (Speech
Commands, LibriSpeech incl. dev-other / test-other) with deterministic
code augmentation; wav2vec2 used as a locating tool only. Nothing
published.

## v4 exam freeze (recorded after merge, before training)

- Exam cache `cacheV4_test.npz` frozen on the training runtime:
  **sha256 51d958c8824b741bdd2cbc550d87982fb7a41d626faac5710c02e365ea63123e**,
  8,475 clips from LibriSpeech test-other (speakers per
  `test_manifest_v4.json`), `_other_` 800, `_silence_` 300.
  Disjointness verified at merge (exam ∩ pool / val / v3-test-clean
  speakers all empty after pre-freeze drops; see manifest).
- Validation cache `cacheV4_val.npz` (LibriSpeech dev-other): 7,344
  clips, frozen in the same merge pass.
- Mining outcome: all 50 words mined from continuous speech (the 16
  former Speech Commands words now have LibriSpeech cuts — `on`,
  `one`, `two`, `no`, `go`, `down` all at the 800 cap; `cat` 66,
  `dog` 85, `bed` 270, `bird` 77). `_other_` pool: 14,000 clips,
  5,419 distinct words. `jump` (14 clips) fell below the ≥40 bar and
  was dropped per the registered rule → **49 vocab classes**, as v3.

## Results (finalize_v4.py — run exactly once, 2026-11-19)

**Verdict: v4 REJECTED — v2 stays keeper.** 2 of 5 pre-registered
gates passed. No re-tuning against the fresh exam was done or will be
done; the exam is spent for this design.

Execution note: the run took a queue wait (behind Write It Right v6),
then a driver death (VM restart) after the merge phase; the surviving
runtime was adopted and the remaining phases (train → finalize) were
driven by a recovery pass, with finalize run exactly once. Training:
28 epochs on T4, best checkpoint epoch 27, validation accuracy
**0.8192** (dev-other — and unlike v3, validation predicted the exam
closely). Compute: **1.84 units** of the 8-unit cap (balance
590.80 → 588.96 over the job window). All artifacts were pulled back
and verified locally (byte-size match + ONNX load checks) before the
runtime was deleted; deletion verified by list_runtimes afterwards.

Decision rule tuned on **validation only**: τ = 0.30, margin = 0.0
(validation FA 0.0027, true-accept 0.8134). The fresh exam
(LibriSpeech test-other, 8,475 clips, 33 speakers, sha256 51d958c8…)
was then evaluated **once** (fp32 + int8 in the same pass):

- **Test overall accuracy 0.8031** (gate ≥ 0.93 ✗); **vocabulary-word
  accuracy 0.8072** (gate ≥ 0.93 ✗). Per source: test-other 0.7962,
  silence 0.9900.
- **False-accept (v1 argmax definition): 0.2245** (gate ≤ 0.05 ✗) —
  a 3.2× improvement on v3's 0.7191, still far from the gate. With
  the tuned decision rule: FA 0.0038 on `_other_`/`_silence_` clips
  (0.0029 over all pairs) at a true-accept of 0.7981.
- **Coverage 49 words ✓** (only `jump` below the ≥40-clip bar).
  **int8 ONNX 173,185 B ✓** (fp32 605,236 B; 151,955 params). int8
  test overall 0.7992.
- **The distribution fix worked on its target.** The words that
  collapsed in v3 recovered: `on` 0.004 → **0.760**, `one` 0.071 →
  **0.862**, `two` 0.030 → **0.883**, `no` 0.155 → **0.871**, `go`
  0.333 → **0.902**, `three` 0.361 → **0.857**, `down` 0.574 →
  **0.848**. `_other_` argmax accuracy 0.016 → **0.695**.
- What binds now is different: broad mid-range accuracy on noisy
  continuous speech (`the` 0.680, `it` 0.727, `in` 0.773, `little`
  0.554, `look` 0.636, `a` 0.686), plus a few tiny-n picture words
  (`dog` 0.222 on n=9, `bird` 0.0 on n=1, `cat` 0.667 on n=3 — read
  with care). Note the exams are not cross-comparable: v3's was
  test-clean, v4's is the noisier test-other partition, chosen so
  validation and exam conditions match.
- Lesson for any v5: the word-distribution problem is fixed; the
  remaining gap is robustness/capacity on "other"-condition speech.
  Levers not yet tried: the unused int8 headroom (173 KB of the
  0.5 MB cap) for a larger model, more training data (train-clean-360
  / train-other subsets), and augmentation matched harder to
  "other" recording conditions. A fresh exam would be required,
  since this one is spent.

Artifacts retained locally for the record: `metrics_v4.json`,
`model_v4.onnx`, `model_v4_int8.onnx`, `classes_v4.npy`,
`cacheV4_test.npz`, `ls_clips_v4.npz`, `ls_other_v4.npz`,
`ckpt_v4/` (best.pt = epoch 27), phase logs, manifests.
v2 remains the shipped model; nothing published.

## Terms safety (v4)

Same rules as v1–v3, unchanged: no Muse or OpenAI/Luna output in
data, labels, selection, or filtering; only CC BY 4.0 corpora
(Speech Commands, LibriSpeech incl. dev-other / test-other) with
deterministic code augmentation; wav2vec2 used as a locating tool
only. Nothing published.

---

# v5 (2026-11-19) — PRE-REGISTRATION (written before v5 training and
# before the v5 exam was built or evaluated)

**Validation label (unchanged): validated on adult/teen voices, not yet
on children 5–8.**

**Design basis — v4's diagnosis.** v4's continuous-speech distribution
fix worked (`on` 0.004→0.760, `one` 0.071→0.862, `two` 0.030→0.883,
`_other_` argmax accuracy 0.016→0.695; argmax false-accept
0.7191→0.2245, and 0.0038 under the validation-tuned rule). What binds
now is broad robustness on noisy ("other"-condition) speech — `the`
0.68, `it` 0.73, `in` 0.77, `little` 0.55 on v4's test-other exam —
recorded in v4's results as a capacity/data-scale problem, not a
distribution one. **v5 keeps v4's data design** (continuous-speech
mining for every word + large diverse `_other_`) and changes only
scale, capacity, and augmentation hardness. v4's exam is spent and is
never re-used; v2's and v3's exams likewise.

## v5 design (all registered here, before any v5 run)

1. **Data scale-up — new pool from LibriSpeech train-other-500**
   (OpenSLR 12, CC BY 4.0), a partition no prior version has touched
   (1,166 speakers, "other"-condition acoustics — the register that
   binds). Mining mirrors v4's method (CTC forced alignment against
   the corpus's own human transcripts; wav2vec2 as locating tool
   only): all 50 vocab words, cap 1,400/word; non-vocab words as
   `_other_` candidates (alphabetic, length ≥ 2, cap 30 per distinct
   word, 26,000 total). Utterances ≤ 12 s, plus 12–30 s utterances
   containing a target word. Mining stops early once all caps are
   reached. train-clean-360 was considered and NOT used: it is
   clean-condition, and the binding constraint is noisy-condition
   robustness; train-other-500 is the on-target scale-up. The v5 pool
   is the union of this new mining and v4's pool (dev-clean +
   train-clean-100 continuous-speech clips, reused as training
   material only).
2. **Exam speakers are carved out of train-other-500 first.** A
   speaker is an EXAM speaker iff md5(speaker id) % 25 == 0
   (deterministic, ~4% of speakers). Exam speakers are excluded from
   all v5 training mining. Training speakers from train-other-500 are
   all remaining speakers. Disjointness is additionally verified in
   the merge step against: v5 training-pool speakers, v5 validation
   speakers (dev-other), v3's spent test-clean speaker list, v4's
   spent test-other speaker list, and v2's test assignment (pool
   speakers with md5%10 == 9, plus Speech Commands testing_list —
   neither is used as exam material here). Any overlap found is
   dropped before freezing and recorded in the exam manifest.
3. **Fresh exam — held-out train-other-500 speakers** (same
   "other"-condition register as v4's exam, so the v4→v5 comparison
   is meaningful). Extraction mirrors v4's exam procedure: all-word
   CTC extraction from exam speakers only, vocab cap 600/word,
   `_other_` candidates cap 30/distinct word and 1,200 total, sampled
   to 800 with seed 5150; `_silence_` = 300 fresh slices of the Speech
   Commands background-noise recordings, seed 3105 (disjoint in
   procedure from seeds 11 / 777 / 2026 / 2027). The exam cache and
   its manifest (per-word counts, speaker list, sha256) are frozen
   and recorded before training starts; the exam is evaluated
   **exactly once**, in the finalize pass.
4. **Validation — LibriSpeech dev-other, rebuilt from v4's extracted
   dev-other clips** with the v5 class list (same procedure as v4's
   validation build: class words keep labels, non-vocab → `_other_`
   sampled to 800 with seed 98, vocab-not-class excluded; `_silence_`
   = v4's seed-2027 slices). Checkpoint selection and decision-rule
   tuning use this validation set only.
5. **Class rule (unchanged):** a word is a class iff it has ≥ 40 clips
   in the combined v5 pool; words below the bar are dropped, never
   padded (v4: 49 classes, `jump` dropped at 14 clips).
6. **Training-set construction (v4's rules, scaled):** LibriSpeech
   clips primary — train-split speakers only (v4 pool keeps its
   md5%10 split: train ≤ 7, == 8 unused, == 9 excluded; train-other
   keeps the exam carve-out above) — base cap **1,000/word** (v4:
   600). Speech Commands secondary, capped per word at that word's
   LibriSpeech train base count; SC `_other_` cap 2,000; SC
   `_silence_` cap 900; no variants for SC clips. LS `_other_`: base
   cap **14,000** (v4: 8,000), each base clip + 1 deterministic
   variant.
7. **Augmentation, extended toward "other"-condition acoustics**
   (deterministic, code-only; v4's recipe extended, not replaced):
   each LS vocab base clip gets 3 variants — (a) speed 0.9/1.1
   (v4), (b) noise mix at SNR drawn 0–18 dB (v4: 5–20 dB; harder),
   (c) reverb + noise combined (new combination of v4's two
   operations). Classes with < 200 base clips get a 4th variant
   (the remaining v4 operation). `_other_` variants alternate
   noise / reverb+noise. In-training SpecAugment-style frequency/time
   masking and time rolls unchanged from v3/v4.
8. **Capacity increase within the int8 gate.** DS-CNN widened:
   stem 128; blocks 128→160, 160→224, 224→288 (stride 2), 288→288,
   288→384 (stride 2); ≈ 350 k params (v4: 152 k). The finalize
   preflight must confirm int8 ONNX ≤ 0.5 MB before training; if it
   exceeds, widths are reduced and the change recorded here before
   training. Training: 36 epochs, AdamW lr 2e-3 cosine, label
   smoothing 0.05, batch 256; checkpoint selection on validation
   accuracy only.
9. **Decision rule** keeps v2–v4's form (accept iff p(word) ≥ τ and
   margin over runner-up ≥ m), tuned on validation only with
   false-accept budget ≤ 0.10, reported alongside — the argmax gate
   below is the gate.

## Pre-registered gates (ALL must pass or v5 is rejected, v2 stays
## keeper, and there is no re-tuning against the fresh exam)

1. Fresh-exam **overall accuracy ≥ 0.93**
2. Fresh-exam **vocabulary-word accuracy ≥ 0.93**
3. Fresh-exam **false-accept ≤ 0.05** (v1 definition: `_other_` /
   `_silence_` clips whose argmax is a vocabulary word)
4. **Coverage ≥ 40 words** (aim: keep v4's 49)
5. **int8 ONNX ≤ 0.5 MB**

## Terms safety (v5)

Same rules as v1–v4, unchanged: no Muse or OpenAI/Luna output in
data, labels, selection, or filtering; only CC BY 4.0 corpora (Speech
Commands, LibriSpeech incl. train-other-500) with deterministic code
augmentation; wav2vec2 used as a locating tool only. Nothing
published.

## v5 outcome (2026-11-19): NO VERDICT — run died at merge; exam never evaluated

The v5 run never reached training. Corpora (train-other-500, 30.6 GB,
gzip-verified, 151,472 files), pool mining, and exam extraction all
completed on the Colab T4 runtime; the merge process then died silently
mid-build — after loading all pool shards (OTHER_POOL 40,000 clips /
11,029 distinct words; CLASSES 52; N_VOCAB 50, DROPPED []) and before
writing cacheV5_train.npz. No traceback in the log; the signature is
consistent with an OOM kill while merge held the full augmented train
set in RAM (v5's in-memory footprint is far larger than v4's). The
driver's finally block then deleted the runtime per protocol, taking
the pool shards, the extracted exam clips (ls_exam_v5.npz), and the
exam manifest with it; none had been pulled locally yet.

Consequences, recorded plainly:
- finalize_v5.py NEVER ran; metrics_v5.json does not exist; the frozen
  exam was never evaluated. No held-out contamination occurred, and the
  pre-registered exam design (md5 speaker rule over train-other-500)
  remains legitimately reusable by a future rerun.
- v2 remains the keeper, untouched.
- Spend: account balance 588.70 -> 578.84 across the job window
  (9.86 units account-wide; the window was shared with another chat's
  runtime, and the v5 T4 itself was alive ~7.1 h incl. two VM-restart
  idle gaps and a ~3.9 h corpus download — roughly the 8-unit cap on a
  runtime-hours basis). The cap is consumed; a rerun needs fresh
  authorization.
- Rerun prerequisites (learned the hard way): (1) merge must stream /
  chunk its augmentation instead of holding all variants in RAM;
  (2) every phase's outputs must be pulled locally as soon as the phase
  completes — rescue pulls wired only into the budget-floor path did
  not cover a plain phase failure; (3) the driver must survive VM
  restarts (the phase-guard resume pattern worked; three relaunches
  were needed).

---

# C4 — child-gap measurement (2026-11-19) — PRE-REGISTRATION (written BEFORE any child audio was ingested and BEFORE any scoring)

**Validation label: validated on adult/teen voices, not yet on children 5–8.**

Consent/permitted-use record: `child_eval/CONSENT.md` (written first,
sources checked at source 2026-11-19). Summary: JIBO Kids — IRB +
parent/student informed consent for collection and distribution of
anonymized speech (Shankar et al. 2024, §6.2); use with attribution per
the DrivenData directory; transcripts CC BY 4.0. Arizona Child Acoustic
Database (ACAD) — University of Arizona IRB #1000000615; guardian
consent explicitly covers databank sharing for future research without
additional consent, identifiers withheld (TalkBank permission form);
TalkBank Ground Rules = CC BY-NC-SA 4.0, local storage only, no
redistribution, no web/frontier processing. C4 use for both: local
evaluation only, noncommercial research, never trained on, never
redistributed. Both CLEARED for that use; handling rules in CONSENT.md.

## Design (fixed in advance)
- Corpora: JIBO Kids (ages 4–7, grade bands PreK/K/G1 as its age proxy)
  + ACAD (ages 2–7 at recording). Age bands reported: 2–3, 4–5, 6–7
  (ACAD by age at recording; JIBO by grade proxy PreK≈4–5, K≈5–6,
  G1≈6–7, disclosed as a proxy wherever reported).
- Vocabulary restriction: only clips whose label is one of v2's 40
  vocabulary words are scored as word clips. Non-vocab isolated words
  (ACAD only, where the label is known and the clip is a single word)
  form the _other_ probe population for false-accept measurement.
  Coverage (which vocab words each corpus actually contains) is
  reported honestly; no word is padded or synthesized.
- Clip extraction: ACAD word files are used as-is. JIBO session audio
  is used ONLY where a transcript provides a time-aligned segment for a
  single vocabulary word; no frontier/forced-alignment output is used
  for labels or selection. If JIBO yields no such segments, that is
  reported as a JIBO coverage result, not worked around.
- Waveform rule (mirrors v2's own eval preprocessing): resample to
  16 kHz if needed; a clip longer than 1 s is center-cropped to the
  1 s around its energy peak; shorter clips are zero-padded to 1 s.
  Mel = v2's exact pipeline (torchaudio MelSpectrogram SR 16000,
  n_fft 512, hop 256, n_mels 40; log(mel + 1e-6)); normalization =
  metrics_v2.json mel_mean/mel_std.
- Split: DISJOINT BY CHILD within each source. Deterministic rule:
  md5(corpus_child_id) mod 10 <= 4 → DEV, else EXAM. No child appears
  in both. EXAM is frozen (manifest with child IDs, per-band and
  per-source counts) and is NOT scored in C4 — it is reserved for the
  G1 final gate.
- Scoring (DEV only): keeper v2 int8 ONNX (model_v2_int8.onnx) +
  v2's shipped verifier rule (accept expected word e iff p(e) >= 0.30,
  margin 0.0 — metrics_v2.json decision_rule). Reported: overall
  argmax accuracy, verifier true-accept on true-word clips,
  verifier false-accept over (clip, expected != true) pairs and on
  _other_ probes, per age band, per source, per word (with n).
  Adult reference: v2 test overall 0.9113 / int8 0.9088,
  true-accept 0.9015 (fp32) — the gap is dev-child minus that.
- No training of any kind. Local CPU only, 0 Colab units. Nothing
  published or redistributed.

## Results (appended after the run)

### C4 results (2026-11-19, local CPU, 0 Colab units)

**JIBO Kids: BUILT AND SCORED.** Audio release (383 session WAVs,
16 kHz) verified against the GitHub LFS SHA-256 for both release
parts. Labels from the DrivenData transcripts (CC BY 4.0, time-aligned
utterances, child-only). Candidates = utterances whose normalized
transcript is exactly one v2 vocabulary word, excluding utterances
flagged `inexact_transcription` (unintelligible/crosstalk): **782
clips, 106 children, 15 of v2's 40 words** (yes, no, one, two, three,
a, and, dog, tree, cat, bird, go, up, to, for, here — heavily skewed:
no 272, three 120, one 115, two 110, a 102; the other 25 vocab words
do not occur as isolated child utterances in JIBO). Split by the
pre-registered md5 rule, disjoint by child (verified: zero overlap):
DEV 398 clips / 53 children, EXAM 384 clips / 53 children — **EXAM
frozen in `child_eval/exam_manifest.json`, never extracted or scored;
reserved for G1.**

**Keeper v2 (int8 + shipped rule) on JIBO DEV — the child gap is
large and it is a false-REJECT gap, not a false-accept gap:**
- Overall argmax accuracy **0.4497** (n=398) vs adult/teen int8 test
  0.9088 → **gap −45.9 points**. Verifier true-accept **0.3894** vs
  adult 0.9015. False-accept over all (clip, wrong-word) pairs stays
  low at **0.0054** (adult test 0.0017): on child speech the keeper
  mostly refuses or mislabels, it does not confidently accept wrong
  words.
- Per grade band (JIBO grade proxy for age): PreK (≈4–5) **0.5333**
  argmax / 0.4667 true-accept (n=60); Kindergarten (≈5–6) **0.3722 /
  0.3158** (n=266); Grade 1 (≈6–7) **0.6667 / 0.5972** (n=72). The K
  band scoring below PreK is reported as measured; word/task mixes
  differ somewhat by band and K carries the largest n, so treat the
  ordering as noisy, not as a developmental claim.
- Per word (dev): one 0.661 (59), two 0.627 (59), no 0.479 (146),
  three 0.453 (64), yes 0.182 (11), **a 0.000 (49)** — 'a' was already
  v2's weakest adult word (0.1429) and collapses entirely on children;
  and 0.000 (5). Top confusions: no→_other_ (20), one→_other_ (14),
  a→on (13), three→tree (13), no→go (11), a→bird (11).

**Arizona Child Acoustic Database: CONSENT CLEARED, ACCESS BLOCKED —
not ingested.** Its isolated-word protocol would have covered 10 more
vocab words (bed, cat, on, dog, go, one, two, three, house, bird,
from the published task list), but: (1) TalkBank's PhonBank media and
CHAT-data downloads for ACAD both sit behind a TalkBank membership
login (verified 2026-11-19 — the endpoints return the login wall to
anonymous access), and creating an account / accepting TalkBank
membership terms is a human step C4 did not take; (2) the University
of Arizona Repository collection (handle 10150/316065) publicly lists
only the Task List and Participant Table items — no audio items.
Unblocking = Rusty (or any team member) signs up for a free TalkBank
membership and downloads the ACAD media, or asks the contributors
(Bunton/Story) for the audio; the consent basis in
`child_eval/CONSENT.md` already covers the planned local eval use,
and the pre-registered split/scoring design above applies unchanged
when the audio lands. Per-source C4 reporting is therefore JIBO-only,
and the frozen EXAM is JIBO-only — G1's child gate will measure the
JIBO distribution unless ACAD arrives before G1 runs.

Deviations/notes: (1) `inexact_transcription` utterances excluded —
a label-quality rule applied before any scoring, recorded here.
(2) JIBO age = grade proxy (PreK/K/G1 from the corpus's own anonymized
IDs), disclosed wherever bands are reported. (3) An implementation
bug in the first scoring attempt (grade-code index) and a slow
energy-window computation were caught and fixed before any results
were read or used; the scored run is `build_score_c4.py` as saved.
(4) Download parts were SHA-256-verified against the corpus's LFS
pointers, then deleted after extraction (383 WAVs verified present);
the extracted release + transcripts stay in `child_eval/`, local only.

Artifacts: `child_eval/CONSENT.md`, `child_eval/dev_manifest.json`,
`child_eval/exam_manifest.json` (FROZEN), `child_eval/metrics_c4_dev.json`,
`child_eval/build_score_c4.py`.

---

# v6 (2026-11-19) — PRE-REGISTRATION (written BEFORE v6 data build,
# BEFORE any v6 training, and BEFORE the C4 child EXAM was extracted
# or scored)

**Validation label (unchanged until a verdict): validated on adult/teen
voices, not yet on children 5–8.**

**Lineup item G1.** Diagnosis it answers (C4, measured 2026-11-19):
keeper v2 on JIBO Kids DEV child speech scores argmax **0.4497** /
verifier true-accept **0.3894** vs 0.9088 / 0.9015 on adult/teen test —
a ~46-point child gap whose failure mode is false REJECTION /
mislabeling (false-accept over all wrong-word pairs only 0.0054), not
false accepts. Per grade band (DEV): PreK 0.5333, K 0.3722, G1 0.6667.
Weakest DEV words: a 0.000 (n=49), yes 0.182, no 0.479, three 0.453
(three→tree a top confusion).

## The one lever

**Vocal-tract-length perturbation (VTLP) of the v2 training data —
nothing else changes.** Rationale: children 5–8 have shorter vocal
tracts, so their formants sit systematically higher than adults'; v2's
recipe contains only crude resampling pitch copies (baked into the
Speech Commands side of the v1/v2 caches at factors 1.2/1.3/1.4 on 60%
of SC words — duration-changing, SC-only, none on the LibriSpeech
side). VTLP warps the spectral envelope toward a shorter vocal tract
while preserving duration, and v6 applies it to BOTH corpora.

- **Data: the existing license-logged v2-recipe corpora only**, in
  their frozen cache form (`cacheV2_train.npz`): (1) Google Speech
  Commands v0.02, CC BY 4.0 (LICENSE verified inside the archive);
  (2) LibriSpeech dev-clean + train-clean-100 clips (OpenSLR 12,
  CC BY 4.0) cut by CTC forced alignment against the corpora's own
  human transcripts (wav2vec2 as locating tool only). No new source,
  no new clips, no child-corpus audio in training (JIBO/ACAD are
  eval-only, per C4 consent record). Classes unchanged: `classes_v2.npy`
  (42 classes, 40 words + `_other_` + `_silence_`).
- **Warp (deterministic code only):** standard piecewise-linear VTLP
  on the log-mel frequency axis of each cached spectrogram (40 mel
  bins, HTK mel scale, f 0–8000 Hz, breakpoint F_hi = 4800 Hz):
  output bin at frequency f samples the input spectrum at the
  inverse-warped frequency, linear interpolation over mel-bin centers.
  Warp factor alpha per clip drawn from {1.15, 1.25, 1.35} with
  numpy RandomState seed **20261119**, clip order = cache order.
- **Training-set construction:** every non-`_silence_` training clip
  (37,769-clip cache: vocab words + `_other_`) appears TWICE — the
  original plus one VTLP-warped copy. `_silence_` clips appear once,
  unwarped (background noise has no vocal tract). The build streams in
  chunks of 1,024 clips into a memmap and never holds the augmented
  set in RAM (v5 OOM lesson); the finished cache is pulled/kept
  locally before any runtime work (v5 phase-pull lesson).
- **Everything else is v2's recipe verbatim:** same DS-CNN
  architecture (42,586 params), 16 epochs, AdamW lr 2e-3, weight
  decay 1e-4, cosine T_max=16, label smoothing 0.05, batch 256,
  in-training SpecAugment-style frequency/time masking + time rolls,
  LibriSpeech oversample x2 (the train script's LS_OVER mechanism,
  applied to the v6 cache's source array), checkpoint selection =
  best accuracy on the UNWARPED `cacheV2_val.npz`. Mel normalization
  mu/sd computed on the v6 train set by the training script, as in v2.
- **Decision rule: v2's shipped rule, frozen** — accept expected word
  iff p(word) >= tau = 0.30, margin 0.0. NOT retuned on any set.

## Evaluation sets (frozen)

- **Child EXAM (final gate, scored ONCE):** the C4-frozen JIBO Kids
  exam split — 384 clips / 53 children, disjoint by child from DEV,
  manifest `child_eval/exam_manifest.json` (frozen in C4, never
  extracted or scored before this version's finalize). Extraction
  (waveform -> mel, C4's exact `build_score_c4.py` procedure:
  16 kHz, 1 s energy-peak crop / zero-pad, torchaudio MelSpectrogram
  n_fft 512 hop 256 n_mels 40, log(mel+1e-6)) computes features only —
  no model inference at extraction time. The exam mel cache stays on
  this machine; finalize scoring is LOCAL (int8 ONNX, CPU), so child
  audio/mel never leaves local storage, is never redistributed, and
  is never trained on. Scored exactly once, in the finalize pass,
  for v2 int8 (baseline) and v6 int8 in the same pass, each model
  normalized with its own shipped mu/sd (v2: metrics_v2.json
  mel_mean/mel_std; v6: its checkpoint's mu/sd), C4 protocol
  (argmax accuracy, verifier true-accept at tau 0.30, false-accept
  over all (clip, wrong-word) pairs).
- **Adult test:** `cacheV2_test.npz` under the metrics_v2.json
  protocol (the parent task's named non-regression set for G1).
  v2 int8 reference: overall 0.9088.
- **Adversarial probes (named in advance, report-only):**
  (i) child-exam clips of v2's weakest DEV words — **a, yes, no,
  three** — reported per word from the same single exam scoring,
  including the three→tree confusion count;
  (ii) **adult VTLP probe:** the first 1,000 clips of cacheV2_test
  warped at fixed alpha = 1.25 (same warp code), scored with v2 int8
  and v6 int8 — checks explicitly that warped adult speech is handled
  (v6 should not collapse on the direction it was trained toward).

## Pre-registered gates (ALL must pass or v6 is rejected, v2 stays
## keeper, no re-tuning against the exam, no same-lever retry today)

1. **Child EXAM improvement (BOTH metrics):** v6 int8 argmax >=
   v2 int8 argmax + 0.05 AND v6 int8 true-accept >= v2 int8
   true-accept + 0.05 on the frozen child exam.
2. **Adult non-regression:** v6 int8 overall accuracy on
   cacheV2_test >= 0.8988 (v2 int8 0.9088 minus 1 point).
3. **Size:** int8 ONNX <= 0.5 MB.
4. **Child false-accept:** v6 false-accept (all-pairs, C4
   definition) on the child exam <= v2's + 0.01.

Per-age-band (PreK / K / G1 grade-proxy bands) and per-word child
exam results are reported beside the headline, per the lineup.

## Compute / budget

Google Colab, ONE GPU runtime, T4 first (fallbacks: T4 highmem ->
L4 -> A100). ONE staging bundle up (v6 train cache + v2 val cache +
classes + training script), checkpoints + logs pulled back before
runtime deletion; runtime deleted in a finally block and verified
via list_runtimes. Balance recorded before/after. Estimate <= 2
units; **hard stop: this job must not spend more than 4 units**
(floor = balance-before - 4.00); day cap 12 units total stands.

## Terms safety (v6)

Same rules as v1–v5, unchanged: no Muse / OpenAI / Luna / Gemini
output in data, labels, selection, or filtering; only the two
CC BY 4.0 corpora above with deterministic code augmentation;
wav2vec2 used as a locating tool only (in v2's prep, unchanged).
JIBO Kids: eval-only under the C4 consent record (IRB + parental
consent for distribution of anonymized speech; attribution), local
use only, never trained on, never redistributed. Nothing published.

## v6 results (2026-11-19): REJECTED — one gate missed narrowly; v2 stays keeper

Executed exactly as pre-registered. Build (`build_cache_v6.py`,
local, streaming/memmap): v6 train cache = 74,638 clips (37,769 v2
originals + 36,869 VTLP copies; the 900 `_silence_` clips unwarped,
originals only), alpha counts 1.15: 12,265 / 1.25: 12,343 / 1.35:
12,261, sha256 be869203d261b2d1a5e229cab654345b871db92191facf348429a6e7eaae3b11.
Child exam extracted locally (`build_exam_v6.py`, C4 procedure,
features only — no inference at extraction): 384 clips, mel
(384, 40, 63), `cacheV6_childexam.npz`, kept local. Training on
Colab T4 highmem (T4 standard had no capacity): 16 epochs, ~150 s
GPU, best checkpoint epoch 15, validation accuracy **0.9239**
(v2: 0.9176). Checkpoints + log pulled and verified locally BEFORE
runtime deletion; runtime deleted, list_runtimes verified empty.
**Cost: 0.02 Colab units (balance 578.84 -> 578.82).** Finalize
(`finalize_v6.py`) ran ONCE, locally: the frozen child exam was
scored in that single pass for v6 int8 and v2 int8.

### Child EXAM (JIBO Kids, n=384, 53 children; int8, frozen rule tau 0.30)

| Metric | v2 int8 | v6 int8 | Delta | Gate |
|---|---|---|---|---|
| Argmax accuracy | 0.4427 | 0.4844 | **+0.0417** | >= +0.05 **FAIL** |
| True-accept | 0.4036 | 0.4609 | +0.0573 | >= +0.05 pass |
| False-accept (all pairs) | 0.0064 | 0.0057 | -0.0007 | <= +0.01 pass |

Per grade band (argmax v2 -> v6; true-accept in parentheses):
PreK (n=74) 0.3784 -> 0.4324 (0.3514 -> 0.4189); K (n=248)
0.4355 -> 0.4798 (0.3911 -> 0.4637); G1 (n=62) 0.5484 -> 0.5645
(0.5161 -> 0.5000). Gains concentrate in the younger bands, as the
lever intended; G1 is nearly flat.

Per word (argmax, n >= 9): three 0.4643 -> **0.6607** (n=56) with
the three->tree confusion down 0.2500 -> 0.1607; one 0.6250 ->
0.7143 (56); dog 0.4444 -> 0.5556 (9); no 0.4683 -> 0.4921 (126);
two 0.5882 -> 0.5882 (51); **yes 0.6842 -> 0.4737 (19) — a real
regression on a small n**; **a 0.0000 -> 0.0000 (53) — VTLP does
nothing for 'a'**: the word is a single short vowel whose child
failure is not a formant-location problem at this scale.

### Adult test (cacheV2_test, metrics_v2 protocol)

v6 int8 overall **0.9177** vs v2 int8 0.9088 — an improvement of
+0.89 points, not merely non-regression (gate floor 0.8988: pass).
v6 fp32 0.9220 (v2 fp32 0.9113). Per source (int8): Speech Commands
0.9232, LibriSpeech 0.8590. Adult rule stats (int8): false-accept
0.0014, true-accept 0.9084.

### Adversarial probes (report-only, as pre-registered)

- Weak-word child probes: covered per word above (a, yes, no, three).
- Adult VTLP probe (first 1,000 test clips warped at alpha 1.25):
  v6 int8 argmax **0.900** vs v2 int8 0.826 — v6 handles
  formant-shifted adult speech far better, confirming the warp
  direction was learned as intended.

### Gates — 4 of 5 pass, v6 REJECTED

1. Child exam argmax +5: **FAIL (+4.17)**; true-accept +5: pass (+5.73).
2. Adult non-regression: pass (0.9177 >= 0.8988).
3. int8 size: pass (62,084 B).
4. Child false-accept: pass (-0.0007).

**Verdict: v6 is REJECTED and v2 remains the keeper**, per the
pre-registered all-gates rule — the argmax gate missed by 0.83
points. No re-tuning against the exam was done; the exam is now
spent for this design. Honest reading: VTLP is a real but partial
lever — it bought +4 to +6 points on children and +0.9 on adults at
0.02 units, with the biggest wins exactly where formants should
matter (three/tree, one, younger bands), but it does not touch the
'a' collapse (0.000 on 53 exam clips for both models) and it cost
'yes' on a small sample. Per the lineup stop rule, no same-lever
retry today; a future child-speech version needs a fresh exam and
a different or additional lever aimed at the residual failure modes
(short function words, 'a'), not a VTLP re-roll.

Artifacts: `model_v6.onnx` (170,937 B), `model_v6_int8.onnx`
(62,084 B), `metrics_v6.json`, `ckpt_v6/` (best.pt = epoch 15),
`cacheV6_train.npz`, `cacheV6_childexam.npz` (local only, eval
material — never redistributed), `cacheV6_adultprobe.npz`,
`build_cache_v6.py`, `build_exam_v6.py`, `train_v6_gpu.py`,
`finalize_v6.py`, `train_v6.log`; driver + results in
`~/workspace/colab/ra-v6/`. Nothing published.

---

## Phase 1 of PLAN-VIEWS-SPECIALISTS (2026-11-19, local CPU, 0 units)

Pointer entry — full record in `PLAN-VIEWS-SPECIALISTS.md` (Phase 1
results section); metrics in `metrics_view3.json`, `metrics_s1.json`,
`metrics_s2.json`, `metrics_mine_a.json`, `metrics_s3.json`,
`metrics_s3b.json`, `metrics_assembly_dev.json`. Child numbers are
JIBO DEV (development) only; the spent C4/G1 exam was not touched.
Headlines: View 3 (inverse-VTLP input warp, α=0.87) gives child dev
argmax +7.3 / true-accept +8.6 at zero training cost. S1 rescue
verifier built (5.7 KB): dev true-accept +3.0 pts vs +18.1 oracle,
child FA 0.0054→0.0064, adult val non-regression holds even ungated.
S2 three/tree built (2.8 KB): dev corrected 2 / broke 0; adult val
corrected 11 / broke 1. 'a' mining fixed (min-span filter was the
scarcity cause): 960 new clips / 221 speakers — but the S3
target-conditioned 'a' verifier does NOT transfer adult→child
(dev AUC 0.57, TA('a') still 0.000) and is rejected as a design;
'a' must be fixed inside a future whole-model retrain. v2 remains
the keeper; a v7 still needs a fresh child exam (ACAD — Rusty's
TalkBank signup) before any child keeper verdict. Nothing published.

---

# Release note — Phase 1 development push (2026-11-19)

- **Repo:** RustyDrops/Read_Aloud_Coach_Tiny_ML (public), branch `main`.
- **Commit:** `411203102ef9f94b15768cf6c39cf33bc204b0f1` — "Read-Aloud Phase 1 (dev): View 3 input-warping +8.6 child true-accept on dev; v2 remains keeper", layered on the v2 release commit `2d81c737ddb5aa53689eb58b344d192f6353eed7`. Pushed at Rusty's instruction ("push the winners").
- **Nature of the push:** DEVELOPMENT update only. v2 remains the keeper and the only released model; the README's new Phase 1 section is labeled "DEVELOPMENT (child DEV numbers only, not a release)" and states plainly that no v7 exists, no new keeper is claimed, and a v7 cannot be crowned without a fresh child exam.
- **Contents (18 files, staged in `release_phase1/`):** README.md (v2 release content intact + the new labeled Phase 1 section), PLAN-VIEWS-SPECIALISTS.md (full), PROVENANCE.md (full local copy as of the push), view3_normalized.py, train_specialists.py, assembly_dev.py, mine_a_tc100.py, s1_model.npz (5,682 B; sha256 7f4e4783…), s2_model.npz, metrics_view3.json, metrics_s1.json, metrics_s2.json, metrics_s3.json, metrics_s3b.json, metrics_mine_a.json, metrics_assembly_dev.json, child_eval/phase0_diagnostics.py, child_eval/phase0_results.json.
- **Verification (read-back):** newest commit matches; all 18 pushed files' Git blob SHAs match local exactly (incl. s1_model.npz and metrics_s1.json); all pre-existing v2 release files (model_v2_int8.onnx, predict.py, metrics_v2.json, promo image, training scripts) confirmed still present.
- **Exclusions:** NO child audio, NO JIBO/ACAD data or clips, no caches/logs/datasets. `child_eval/exam_manifest.json` and `child_eval/dev_manifest.json` were EXCLUDED because they contain per-clip child IDs and audio paths into the restricted JIBO release; only aggregate Phase 0 numbers (phase0_results.json) and the diagnostics code were pushed. Child corpora remain eval-only, local, never redistributed.

---

## Elephant-Goldfish review of the views + specialists round (2026-11-19)

EG review done — full record in `~/workspace/micro-edge-pilot/EG-REVIEW-2026-11-19.md`
(35 findings: 29 accepted, 6 accepted-with-changes, 0 rejected; Luna cost
$0.006224; feasibility lens run by the elephant after two goldfish
disconnects, logged in the review). Amended next round for this model:
**v7 DEVELOPMENT candidate** — single lever = training-data repair (the
960 mined 'a' clips + min-span fix, function-word mining on the same
fix); View 3 (alpha 0.87, now a disclosed design constant), S1, and S2
carried as frozen ablatable components; JIBO DEV quarantined as
development-only; gates fully operationalized in the review §4 and
evaluable only on a NEW child exam (ACAD — Rusty's TalkBank signup).
No keeper verdict is possible until that exam exists. The bolt-on 'a'
verifier design is permanently dropped.

---

# v7 (2026-11-19) — PRE-REGISTRATION (written BEFORE v7 mining,
# BEFORE the v7 cache build, and BEFORE any v7 training or scoring)

**DEVELOPMENT CANDIDATE — NOT A KEEPER CANDIDATE.** Standing statement,
which every v7 artifact carries: **"v7 is a development model; no keeper
verdict exists or is implied until a fresh child exam (ACAD or
equivalent) is scored."** v2 remains the keeper whatever the development
numbers below say. Nothing is published or pushed to GitHub.

**Validation label (unchanged): validated on adult/teen voices, not yet
on children 5–8.**

**Design basis.** EG review 2026-11-19 §4 (Read-Aloud) + findings F4/F5,
and PLAN-VIEWS-SPECIALISTS.md Phase 1 results. Phase 1 established:
(i) View 3 (inverse-VTLP input warp, alpha = 0.87) is the largest child
gain found (+7.3 argmax / +8.6 true-accept on JIBO dev, zero training);
(ii) the bolt-on 'a' verifier (S3) is a dead end — adult-trained
features saturate at ~0 on child clips — and the mined 'a' data must
enter through a whole-model retrain instead (EG F10 closes the sidecar
line permanently); (iii) the 'a' scarcity was a pipeline bug: the
v1/v2 extractor's 0.04 s minimum-span filter silently dropped 1-2-frame
schwas (>99% of 'a' instances), and the Phase 1 fixed-pipeline mining
recovered 960 'a' clips from 221 LibriSpeech train-clean-100 speakers
vs 80 in the v2 pool. This version is that retrain.

## The one lever

**Training-data repair only: the min-span fix in the alignment/mining
pipeline, applied to rebuild the training pool.** Two components, both
extracted with the Phase 1 fixed pipeline (mine_a_tc100.py lineage:
CTC forced alignment against LibriSpeech's own human transcripts,
wav2vec2 as locating tool only; min span 0.01 s — was 0.04; context
pad +-0.08 s — was +-0.04; span cap 1.15 s and the span-repair rule
unchanged from the v2 pipeline):

- **Component A — the Phase 1 'a' mining, folded in as-is:**
  mine_a_shard_*.npz, 960 'a' clips, 221 speakers, LibriSpeech
  train-clean-100 (CC BY 4.0), from utterances the v2 pipeline never
  processed. Record: metrics_mine_a.json.
- **Component B — function-word mining under the same fix**
  (mine_func_tc100.py, written for this version): LibriSpeech
  train-clean-100 utterances NOT previously processed (not in
  ls_done_tc.txt or mine_a_done.txt), <= 12 s, transcript containing
  at least one of the 14 function words below; processed in a fixed
  deterministic order (count of starved-word tokens {we, me, my, up,
  can} descending, then duration ascending); per-word cap 4 instances
  per utterance; time budget 45 minutes (a bounded pass, the project's
  established pattern — v2's own tc100 top-up was bounded at
  3,000/7,727 utterances). Kept words (function words whose short,
  often-reduced forms the 0.04 s filter bites): and, can, for, in, is,
  it, me, my, not, the, to, up, we, you. 'a' is excluded (Component A
  covers it); content words (away, come, here, little, look, make,
  said, see, where) are excluded — their spans are longer, the bug
  barely touches them, and adding them would be data scale-up, a
  different lever. Per-word yields and the span < 0.04 s subset (the
  instances the old filter provably dropped) are recorded in the
  results.

**Disclosed limits of the repair.** (1) Dev-clean is NOT re-extracted:
its 2,362 utterances were already processed by the buggy pipeline, and
re-extraction would duplicate word occurrences already in the pool
(same instance, different padding). Instances the bug dropped in
dev-clean stay dropped; the repair is train-clean-100-only, exactly as
Phase 1's 'a' lever was. (2) Component B is a bounded pass, not an
exhaustive one; its yield is whatever the budget produces, recorded
as measured.

## Pool rebuild (merge procedure verbatim from merge_v2.py)

- Pool = the consolidated v2 pool (ls_clips_v2.npz waves) + Component A
  + Component B clips. Per-word cap 600 by seeded shuffle
  (RandomState 123, merge_v2.py's procedure and iteration order).
- Speaker split unchanged: LibriSpeech speaker id md5%10 — train <= 7,
  val == 8, test == 9. **Only the train split is written**
  (cacheV7_train.npz, same format: mel float16 / label / source).
  Validation for checkpoint selection and all adult reporting is v2's
  frozen cacheV2_val.npz, UNCHANGED — this keeps the adult comparison
  same-set against v2's val numbers. New clips from val-split speakers
  are used nowhere; clips from test-split (== 9) speakers are excluded
  entirely, per the standing v2 discipline (Component A's 221 speakers
  span all buckets: 178 train / 22 val / 21 test — the split rule, not
  hand-picking, decides what trains).
- Class list must equal classes_v2.npy (42 classes: 40 words +
  _other_ + _silence_); the build asserts this and stops if it fails.
  (It cannot legitimately change: A and B add only existing class
  words, and no dropped word is in either keep set.) Speech Commands
  side of the merge is byte-identical to v2's (cache_sc_train.npz
  remapped as in merge_v2.py).
- Architecture, recipe, and vocab follow v2 with NO other change:
  same DS-CNN (42,586 params), 16 epochs, AdamW lr 2e-3, weight decay
  1e-4, cosine T_max = 16, label smoothing 0.05, batch 256,
  LibriSpeech oversample x2 (the train script's source-array
  mechanism), in-training SpecAugment-style masking + time rolls,
  mel normalization mu/sd computed on the oversampled train array by
  the training script, checkpoint selection = best accuracy on
  cacheV2_val. Training script: train_v7_gpu.py = train_v2_gpu.py with
  only the train-cache filename, checkpoint dir (ckpt_v7), and done
  marker changed.
- **Decision rule: v2's shipped rule, frozen** — accept expected word
  iff p(word) >= tau = 0.30, margin 0.0. NOT retuned on any set (v6
  precedent).
- **Compute (disclosed execution choice, not a recipe change):**
  local CPU measures ~830 s/epoch for this recipe (train_v2.log:
  ep 0-3 at 819-3352 s cumulative) -> ~3.7 h for 16 epochs, so training
  runs on Google Colab T4 per the v2/v6 precedent (v2: 0.14 units;
  v6: 0.02 units). Balance before: 576.05 units, 0 active assignments
  (recorded 2026-11-19 evening). **Hard cap for this job: 2 units**
  (floor = balance-before - 2.00); runtime deleted in a finally block
  and verified with list_runtimes; checkpoints pulled and verified
  locally before deletion (v5/v6 lessons). Export (fp32 + int8 ONNX,
  finalize_v2.py's procedure) and ALL evaluation are local.

## Carried frozen components (NOT levers; ablated, never re-tuned)

- **View 3 input warp, alpha = 0.87 — a disclosed design constant**
  (EG G4 disposition). It was selected from a 4-point sweep
  {1.00, 0.87, 0.80, 0.74} on the same JIBO dev that reports it, so
  dev carries selection optimism of unknown size. It is carried
  frozen; alpha is not re-tuned here. The EG G4 nested-CV diagnostic
  (5 child-grouped folds of JIBO dev; alpha selected inside folds
  only, by true-accept, from the same 4-point grid; frozen v2 int8)
  is run and reported as expectation-setting only — it is not
  evidence for the candidate.
- **S1 rescue verifier** (s1_model.npz; threshold 0.425 frozen on its
  Phase 1 holdout; trigger = target in base model's top-2, not
  rule-accepted, median F0 >= 220 Hz) and **S2 three/tree verifier**
  (s2_model.npz; threshold 0.5; trigger = top-2 exactly {three, tree})
  — weights, features, and thresholds frozen AS-IS from Phase 1. In
  the v7 ladder they consume v7's probabilities and mel, exactly as
  assembly_dev.py consumed v2's. Disclosed: S1/S2 were trained on
  v2/v6-era probability distributions, so applying them to v7 outputs
  carries the same distribution-shift caveat Phase 1 recorded for the
  warp assembly. EG section 4's "learned register probe replaces S1's
  raw F0 gate" item is NOT implemented in this run — this brief
  freezes S1 as-is; the probe remains an outstanding productization
  item (the raw gate's ~25% adult-women fire rate is a known blemish;
  Phase 1 showed the system stays safe because the verifier itself is
  strict), recorded here so it is not silently dropped.
- **Dev-use note (EG L4/G5):** JIBO DEV has informed the warp factor,
  the view choices, and the assembly design. It is an adaptive
  development set, used here for development ranking ONLY and
  quarantined from any gating. No claim rests on it. JIBO consent
  basis: C4 record (local evaluation use).

## Evaluation (all local, development only)

**The C4 child EXAM is SPENT and is not touched by this version** —
not extracted, not scored, not used for selection of anything
(child_eval/exam_manifest.json and cacheV6_childexam.npz stay
closed). No new child exam exists; that remains blocked on ACAD
access (Rusty's TalkBank signup) or an equivalent consent-cleared
corpus.

Ablation ladder, scored same-sitting for v7 int8 and (as the
reference) v2 int8, on (a) JIBO child DEV (n = 398 clips / 53
children; child_eval/dev_mel.npz float16 cache, the Phase 1
convention) and (b) adult val (cacheV2_val.npz, n = 6,046):

- Rung 0: base model alone (unwarped input)
- Rung 1: + View 3 (alpha = 0.87 warped input)
- Rung 2: + View 3 + S1 (child dev: F0-gated as designed; adult val:
  UNGATED worst case — the val cache has no waveforms/F0, Phase 1's
  convention, labelled as such)
- Rung 3: + View 3 + S1 + S2

Per rung: argmax accuracy; verifier true-accept (frozen rule, plus
S1 extra accepts at rungs 2-3); false-accept over all (clip,
wrong-word) pairs; S1-corrected argmax (rungs 2-3); S2 corrected /
broke counts (rung 3). Per-word child-dev table at rungs 0 and 3
(argmax + true-accept, with n; esp. a, three, one, no) and per-word
adult-val table for rung 0. Size: v7 int8 ONNX bytes; assembly total
= base int8 + S1 (5,681 B stored) + S2 (2,806 B stored).

## Development targets (NOT gates — no keeper verdict exists)

- Child dev argmax / true-accept vs v2 alone (0.4497 / 0.3894) and vs
  the Phase 1 assembly (S1-corrected argmax 0.5302 / true-accept
  0.4849, on this same dev).
- **'a' child-dev true-accept vs 0.000 (n = 49) — the headline repair
  check.** Per EG G6 this is a named REPORT, not a gate: the rate is
  printed beside the headline either way.
- Adult non-regression: v7 int8 val argmax vs v2 int8 val 0.9127
  (same set; v2's adult test headline is 0.9088 — 0.9127 is its val
  equivalent). Target: within 1 point.
- False-accept <= 0.01 on child dev and on adult val, every rung.
- Any component with a negative dev ablation is reported as such;
  per EG section 4 it would be removed before a candidate freezes —
  but no candidate freezes here, so ablation results are reported,
  not acted on by re-tuning.

## Terms safety (v7)

Same rules as v1-v6, unchanged: no Muse / OpenAI / Luna / Gemini
output in data, labels, selection, or filtering; only the two
CC BY 4.0 corpora (Speech Commands v0.02; LibriSpeech dev-clean /
train-clean-100) with deterministic code augmentation; wav2vec2 used
as a locating tool only — all labels from LibriSpeech's human
transcripts. JIBO Kids: eval-only under the C4 consent record, local
use only, never trained on, never redistributed. Nothing published;
no GitHub push.

**Pre-training amendment to the v7 pre-registration (recorded BEFORE
any Component B data was produced).** Component B's utterance order is
amended: the duration-ascending tiebreak is replaced by a deterministic
md5(path) tiebreak, and the <= 12 s filter is applied lazily at
processing time (one sf.info per utterance when reached). Reason: the
upfront scan (an sf.info over every candidate FLAC, ~28k files) is
I/O-bound on this machine and had produced no output — and no clips,
no shards, no done-file entries — when the first launch was killed
during its scan phase. The substantive part of the order (starved-word
token count descending, so the repair targets are mined first) is
unchanged, as are the extractor, keep set, caps, and budget.

## TalkBank route removed — owner decision (2026-11-19)
Rusty: "do not move on talk bank, remove that." The TalkBank signup / Arizona Child Acoustic Database route is permanently off the table and must not be re-raised. The child-exam problem is therefore OPEN with no current route: future Read-Aloud versions remain development-ranked (JIBO dev + adult data) unless a different consent-cleared child corpus is found.

**Owner-decision amendment to the v7 pre-registration (2026-11-19
22:22 CST, recorded before training).** The TalkBank signup route is
REMOVED permanently. Earlier references in this pre-registration to
ACAD / a fresh child exam as the pending unlock for a keeper verdict
are superseded and must not be presented as a future route anywhere.
Child-exam status, verbatim: "no current route to a fresh child exam;
development ranking only." The standing statement carried by every
v7 artifact is therefore: "v7 is a development model; no keeper
verdict exists or is implied. Child-exam status: no current route to
a fresh child exam; development ranking only." Everything else in
the pre-registration is unchanged.

## v7 DEVELOPMENT retrain — RESULTS (2026-11-19)

**v7 is a development model; no keeper verdict exists or is implied.
Child-exam status: no current route to a fresh child exam; development
ranking only.** v2 remains the keeper. Per Rusty's 2026-11-19 standing
GitHub rule (pass frozen gates AND improve -> push as default): the
rule was considered and v7 does NOT qualify — its keeper gates are
only evaluable on a fresh child exam, an unevaluated gate is not a
passed gate, and no such exam exists. v7 was NOT pushed to GitHub.

### Run record
- Colab: balance 576.05 -> 576.01 = **0.04 units used** (cap 2.0).
  T4 runtime deleted in the driver's finally path; list_runtimes
  verified empty afterwards. (One aborted first attempt: a NameError
  in the driver's own verify cell — the staged data itself verified
  SHA-OK — fixed and relaunched; included in the 0.04.)
- Training: v2 recipe verbatim on cacheV7_train.npz (38,795 clips;
  sha256 a720a32c...); best epoch 15, float val 0.9163; int8 export
  62,084 B (v2: 62,151 B); full assembly with frozen S1+S2 = 70,572 B.
- Data repair as registered: pool 'a' 80 -> 1,040 (Component A: the
  960 mined clips), train 'a' 64 -> 484 after cap/split; Component B
  added 1,158 function-word clips (me +128, my +122, we +110 pools).
  Component B accounting: the span < 0.04 s subset is exactly 0 of
  1,158 — multi-letter function words never align below 2 CTC frames
  (min observed span 0.04 s), so the old filter's damage was
  concentrated on the single-token word 'a'. (Mining took two runs:
  a runtime restart killed the first at 600 utts; the second finished
  the registered 45-minute budget.)

### Child DEV ablation ladder (JIBO dev, n=398; v7 int8; v2 int8
same-sitting reference in parentheses)
- Rung 0 base:        argmax 0.4698 (0.4523), TA 0.4045 (0.3794), FA 0.0053 (0.0056)
- Rung 1 +View3:      argmax 0.4598 (0.5176), TA 0.4095 (0.4724), FA 0.0054 (0.0048)
- Rung 2 +View3+S1:   TA 0.4422 (0.4849), corrected argmax 0.4874 (0.5302), FA 0.0064 (0.0059); S1 rescued 13 true clips (v2: 5)
- Rung 3 +View3+S1+S2: S2 corrected 7 / broke 0 (3/0); combined corrected argmax 0.4950 (0.5302), TA 0.4422, FA 0.0064
- Development targets: v7 ALONE beats the v2-alone targets
  (0.4497/0.3894 recorded; 0.4523/0.3794 same-sitting). The v7 full
  assembly does NOT reach the Phase 1 assembly targets
  (0.5302/0.4849): **View 3 is a negative ablation on v7** (argmax
  -1.0 pt vs v7's own rung 0) although it was worth +6.5 pts on v2.
  S1 and S2 still add positively on top of it. Frozen components are
  reported, not re-tuned, per the pre-registration.
- False-accept <= 0.01 on every rung, both datasets: MET.

### The 'a' headline repair check: NOT MET — and the mechanism is now known
'a' child-dev true-accept is 0.000 at every rung (n=49), unchanged
from v2; argmax also 0.000. Diagnostic (recorded in
metrics_v7_dev.json): v7 fits ADULT 'a' — 90.3% argmax on its own
484 train 'a' clips, mean P(a) 0.792 — so the data repair did fix
the class starvation it targeted. But on child 'a' clips v7 assigns
mean P(a) 0.004 (max 0.028), scattering to on/go/bed/_other_/two,
and predicts 'a' zero times in all 398 dev clips. The remaining
failure is the adult->child register/channel gap for this
ultra-short word — the same wall Phase 1's adult-trained bolt-on
verifier hit — not a training-count gap. More adult 'a' data will
not close it.

### Adult val (n=6,046, frozen cacheV2_val; v2 in parentheses)
- Rung 0: argmax 0.9170 (0.9127) — non-regression MET (+0.43 pt);
  TA 0.9152 (0.9114), FA 0.0014 (0.0016).
- +S1 (ungated worst case): TA 0.9238, corrected argmax 0.9239, FA 0.0018.
- +S2: corrected 3 / broke 4 on v7 (v2: 11/1) — slightly negative on
  v7 adults, reported as-is.
- Warped-adult diagnostic (out-of-design): argmax 0.8784 — the warp
  hurts adults, as expected for a child-register transform.

### Per-word child dev (rung 0 -> rung 3 argmax; TA in metrics file)
a (n=49): 0.000 -> 0.000. three (n=64): 0.359 -> 0.609 (S2 rescue).
one (n=59): 0.661 -> 0.644. no (n=146): 0.555 -> 0.507, TA 0.418 ->
0.425. two (n=59): 0.678 -> 0.712. yes (n=11): 0.182 -> 0.182.
and (n=5): 0.000 -> 0.000.

### G4 nested-CV alpha diagnostic (expectation-setting only)
CV estimate at fold-selected alpha: TA 0.4598 / argmax 0.5075 vs
naive full-dev alpha=0.87: 0.4724 / 0.5176 — selection optimism
~1.3 TA pts. 4 of 5 child-grouped folds selected 0.87 (fold 0: 0.74).

### Pipeline verification
The same-sitting v2 reference reproduces Phase 1's recorded numbers
exactly on child-dev rungs 1-3 and the full adult ladder. Child-dev
rung 0 differs slightly (0.4523/0.3794 here vs C4's 0.4497/0.3894 —
1 clip argmax, 4 clips TA), most consistent with C4 having scored
pre-cache float32 mel; all comparisons above use the same-sitting
reference.

### Still open (unchanged by this run)
- Child exam: no current route to a fresh child exam; development
  ranking only. No keeper verdict is possible for v7.
- EG's learned register probe to replace S1's raw F0 gate was not
  implemented in this run (S1 carried frozen per the brief); it
  remains an outstanding productization item.
- Full numbers: metrics_v7_dev.json; mining: metrics_mine_func.json;
  build: build_v7_meta.json; Colab run: ~/workspace/colab/ra-v7/results.json.

---

# RA-FE1 (2026-11-21) — audio front-end candidate: FAIL, no push

Inference-only candidate (v2 int8 unchanged, sha256 fba22193…; validation
label unchanged: adult/teen only). Design + frozen gates:
`autonomous/ra-fe1/PREREGISTRATION.md`; full verdict:
`autonomous/ra-fe1/VERDICT.md`. FAIL 2/6 gates: adult clean argmax
0.7887 (gate ≥0.8988), dirty pooled worse on both sets. Child dev 'a'
TA stayed 0.000 (no drop, no gain). Ablation: energy trim −12 pts and
per-utterance mel normalization (zeroes adult 'a'/'the'/'we') are the
harmful components; high-pass + RMS leveling alone is near-neutral
(0.9015) and won child-dev gain−12 — logged as future RA-FE2 idea only.
Baseline reproduction was exact (waveform sets rebuilt from source,
6,413 adult + 398 child-dev clips, mel match ≤0.0039). Local CPU only,
0 Colab units. Nothing pushed; shipped predict.py unchanged.

---

# RA-PW1 (2026-11-21) — per-word thresholds: FAIL, no push

Decision-rule candidate (v2 int8 unchanged; label unchanged). Design +
frozen gates: `autonomous/ra-pw1/PREREGISTRATION.md`; verdict:
`autonomous/ra-pw1/VERDICT.md`. Per-word τ tuned on adult val + child
dev only, frozen (`tau_table.json`) before the single adult-test
scoring: test TA 0.8985 (gate ≥0.9077) FAIL; FA passed (0.0016 /
0.0018); child dev 'a' in-sample TA 0.000 (gate ≥0.10) FAIL — child
'a' p-band max is 0.0577 and 'a' is never argmax, so no threshold can
accept it. Global τ=0.30 stays shipped. Local CPU only, 0 Colab units.

---

# RA-SP (2026-11-21) — specialist verifiers, queue #3: SP0 FAIL (closed), SP1 FAIL by 0.00004, no push

Design + frozen gates: `autonomous/ra-sp/PREREGISTRATION.md`; verdict:
`autonomous/ra-sp/VERDICT.md`. SP0 ('a', invariant features only, the
named change vs dead S3): child dev AUC 0.5377, FA 0.1289 — FAIL, and
per preregistration the 'a' specialist line is CLOSED permanently
(third strike). SP1 ('no' rescue specialist, threshold frozen on adult
val): child dev 'no' TA 0.3836→0.6233, adult test TA 0.8988→0.9039,
adult 'no' TA 0.8914→0.9728 — but adult test FA pairs exact 0.0020366
vs gate ≤0.0020: FAIL by 0.00004, no push. Follow-up RA-SP2 logged
(exact-FA scan, rank-2-only gate). Local CPU only, 0 Colab units.

---

# RA-PHRASE (2026-11-21) — phrase-window verification: FAIL, no push

Product-logic prototype (synthetic spliced phrases, dev only; v2
unchanged, label unchanged). Design + frozen gates:
`autonomous/ra-phrase/PREREGISTRATION.md`; verdict:
`autonomous/ra-phrase/VERDICT.md`. Restricted function-word argmax
gives child dev 'a' 0.0408 isolated and 0.000 in spliced phrases;
adult spliced 0.010. FAIL gates 1/3/4, gate 2 unevaluated (n=5).
Local CPU only, 0 Colab units.

---

# RA-SP2 (2026-11-21) — 'no' rescue, FA-exact: FAIL by one clip, no push, line closed

Design + frozen gates (+ one pre-scoring amendment, recorded in the
file): `autonomous/ra-sp2/PREREGISTRATION.md`; verdict:
`autonomous/ra-sp2/VERDICT.md`. Adult gates pass (TA 0.90239, FA exact
0.0017935); child dev 'no' TA 0.45205 vs gate 0.4541 — 66 of 146 clips
where 67 were needed. Per its own preregistration the 'no' rescue
line closes here: SP1 (loose) broke FA, SP2 (strict) misses the
child gate, and a middle setting chosen now would be test-tuning.
Local CPU only, 0 Colab units.

---

# RA-D1 (2026-11-21) — short-word temporal view: FAIL gates 1+2, no push

Design + frozen gates: `autonomous/ra-d1/PREREGISTRATION.md`; verdict:
`autonomous/ra-d1/VERDICT.md`. Waveform reconstruction verified 100%
against frozen caches pre-training. Colab T4, 0.03 units (balance
delta). Adult val 0.9211 int8 — best on record — and child TA rose
to 0.4171, but child 'a' stayed 0.000 (0/49) and child argmax fell
to 0.4548 vs the 0.4698 line. v2 remains default; nothing pushed.

---

# RA-D4 (2026-11-21) — formant/duration aux channel: FAIL gates 1+2, no push

Design + frozen gates: `autonomous/ra-d4/PREREGISTRATION.md`; verdict:
`autonomous/ra-d4/VERDICT.md`. Colab T4, 0.03 units (balance delta).
Child dev regressed vs v7 (argmax 0.4221, TA 0.3719); child 'a' TA
still 0.000 (argmax 0.0204, its first nonzero). Adult val 0.9105 int8
passed. v2 remains default; nothing pushed.

---

# RA-FE2 (2026-11-21) — high-pass + RMS leveling only: FAIL gates 2+4, no push

Design + frozen gates: `autonomous/ra-fe2/PREREGISTRATION.md`; verdict:
`autonomous/ra-fe2/VERDICT.md`. Leveling fixed the gain−12 condition
(adult TA +3.1, child TA +9.6 pts) but child clean argmax slipped to
0.4347 (gate 0.4397) and pooled dirty argmax gained only +0.004 adult /
−0.005 child vs the +0.02 gate. Front-end line closed; shipped
front-end unchanged. Local CPU only, 0 Colab units.

---

# RA-PT1 (2026-11-21) — register-shift augmentation: FAIL gates 1+2, no push

Design + frozen gates: `autonomous/ra-pt1/PREREGISTRATION.md`; verdict:
`autonomous/ra-pt1/VERDICT.md`. Colab T4, 0.06 units (balance delta).
Best adult val on record (0.9300 int8) and best child TA (0.4322),
but child 'a' stayed 0.000 even with pitch-shifted adult 'a' in
training — the 'a' gap is not register. RA-PT2 skipped per its own
precondition. v2 remains default; nothing pushed.

---

# RA-FOC1 (2026-11-21) — short-word class-weighted loss: FAIL gates 1+2+4, no push

Design + frozen gates: `autonomous/ra-foc1/PREREGISTRATION.md`;
verdict: `autonomous/ra-foc1/VERDICT.md`. Colab T4, 0.04 units
(balance delta). Adult val fell to 0.8961 and child 'a' stayed
0.000; a few child short words hit program bests ('no' TA 0.4932)
but headline gates rule. v2 remains default; nothing pushed.

---

# RA-MIX1 (2026-11-21) — mel mixup: FAIL gates 1+2+4, no push

Design + frozen gates: `autonomous/ra-mix1/PREREGISTRATION.md`;
verdict: `autonomous/ra-mix1/VERDICT.md`. Colab T4, 0.01 units
(balance delta; ~0.10 rate estimate). Child argmax hit the line
exactly (0.4698) but TA fell to 0.3794, 'a' TA stayed 0.000, adult
val int8 0.9065 missed by 0.0005. v2 remains default; nothing pushed.

---

# RA-PH2 (2026-11-21) — phrase-window training + stretched 'a': FAIL gate 1 only, no push

Design + frozen gates: `autonomous/ra-ph2/PREREGISTRATION.md`;
verdict: `autonomous/ra-ph2/VERDICT.md`. Colab T4, 0.05 units
(balance delta). First candidate to pass gates 2–5 together (child
argmax 0.4749, TA 0.4070, FA 0.0037, adult val 0.9147) — child 'a'
still exactly 0.000. All-gates rule: no push. v2 remains default.

---

# Campaign close (2026-11-21) — Read-Aloud autonomous iteration day

Fourteen candidates run, four killed pre-test by Elephant Goldfish
critique, one EG modification declined as dev-fitting (RA-LA2).
Nothing passed all frozen gates; nothing was pushed; **v2 + global
τ=0.30 remain the shipped default**, adult/teen validation label
unchanged. Total Colab spend for the day: ~0.22 units (balance
deltas). Luna: EG critiques only, ~$0.002.

Best results on record, all rejected on the all-gates rule:
RA-PT1 adult val 0.9300 int8 (register-shift augmentation);
RA-PH2 first to pass child gates 2–5 together. The child 'a' slice
(0.000 across every candidate) is diagnosed as a label/task
mismatch: JIBO 'a' is a long open vowel in a phrase window; every
licensed training 'a' is a 56 ms article blip, and no transform of
it (duration, context, register, formant means) transfers. Next
step belongs to Rusty: new data, an adult-side product decision on
the PT1 recipe, or accept v2 as-is. Full record: autonomous/
(PREREGISTRATION/VERDICT per candidate, rejects.tsv, DIAGNOSTIC-A.md).

---

# v3 SHIPPED (2026-11-21) — RA-PT1 recipe becomes the release

Per Rusty's direction ("whatever you think is best of the 3
questions"), the PT1 register-shift recipe was finalized as v3 and
pushed as the repo default. Adult test (int8, shipped artifact
verified end-to-end with the v2 input contract — normalized mel):
overall 0.9228 (v2: 0.9088), decision-rule TA 0.9163 (v2: 0.8977),
FA pairs 0.0013 (v2: 0.0017). fp32: 0.9239 / TA 0.9173. Per-source
fp32: Speech Commands 0.9334 (v2 0.918), LibriSpeech 0.8228 (v2
0.837 — a small dip, disclosed in the README table). Decision rule
unchanged and NOT retuned (τ=0.30, margin 0.0). Validation label
unchanged: adult/teen voices only, NOT validated on children 5–8;
child numbers remain JIBO-dev-only (v3 dev: argmax 0.4598, TA
0.4322, 'a' 0.000). model_v3_int8.onnx 62,084 B, sha256
55fc825d4b14677dd97a886cd641cd2d0c2f8f81c0a85bfb30f480fd8925817a.
v2 files remain in the repo.

---

# RA-C1 + RA-W1 (2026-11-21) — combination and warp: both FAIL gate 1 only

RA-C1 (union of register-shift, VTLP, and phrase views; 155,180
rows; Colab T4, 0.09 units): adult val 0.9287 int8, child dev
argmax 0.4874 / TA 0.4397 / FA 0.0039 — gates 2–5 pass; child 'a'
0.000. RA-W1 (frozen alpha-0.87 warp on RA-PH2): child argmax
0.5327 / TA 0.4598 — best headline stack on record; 'a' 0.000;
killed at step 1 per its EG modification. v3 remains default.

# JIBO training approval + yield finding (2026-11-21)

Rusty approved training on JIBO Kids with attribution, reversing
the C4 voluntary eval-only hold (see child_eval/CONSENT.md
addendum). Split audit: dev = 53 children, exam = 53 different
children, zero overlap. Only 4 children (keys 205, 266, 274, 298)
are in neither set. Faithful C4-style extraction (single-word
vocab utterances from DrivenData JSONL + release .txt transcripts)
across all their sessions yields 7 clips total (no x2, yes, two,
three, tree, and one 'a' with a zero-length timestamp) — not a
trainable set. RA-J1 therefore not run. Training on dev or exam
children would destroy the only child evaluations and was NOT
done. Remaining routes (Rusty's call): re-split (train on dev
children, spend the exam as the new evaluation), a new child
corpus (e.g. MyST, needs a data-use agreement), or family
recordings of the 40 words.

---

# v4 SHIPPED (2026-11-21) — RA-J1: trained on the retired dev children, exam-gated

Rusty's re-split decision ("Make it so #1"). RA-J1 = RA-C1 union
cache + the 398 retired-dev JIBO clips x20 (7,960 rows; 163,140
total). Colab T4, 0.09 units. The frozen exam (384 clips, 53
children disjoint from all training children) was scored ONCE, J1
and the C1 control in the same pass:

| Exam (int8) | J1 (v4) | C1 control | v2 (2026-11-19) |
|---|---|---|---|
| argmax | **0.8281** | 0.5104 | 0.4427 |
| rule TA | **0.8203** | 0.4766 | 0.4036 |
| FA pairs | 0.0031 | 0.0051 | 0.0064 |
| 'a' argmax/TA | **0.9057 / 0.9057** | 0.000 / 0.000 | — |

ALL SIX frozen gates passed (autonomous/ra-j1/PREREGISTRATION.md).
Adult test int8 0.9219 (v3: 0.9228), fp32 0.9267 — no adult cost.
Shipped artifact verified end-to-end on the exam with the repo
input contract (0.8281 / 0.8203 / 'a' 0.9057, exact match).
model_v4_int8.onnx 62,084 B, sha256
eb885d4635b628f4546d3cc58e71709d5617d72f59be0bb7555a69556f6ff9fa.
Attribution (JIBO Kids): Shankar et al. (2024) JASA Express
Letters 4(11) 115201; Zenodo 10.5281/zenodo.13964792; DrivenData
transcripts CC BY 4.0. The dev split is retired as an evaluation
from this date; the exam is the sole child evaluation and is
scored sparingly (once per candidate finalize). v3 files retained.

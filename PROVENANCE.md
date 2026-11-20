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

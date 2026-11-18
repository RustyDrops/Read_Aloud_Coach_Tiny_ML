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

## Results

_(to be filled in after the single fresh-test evaluation)_

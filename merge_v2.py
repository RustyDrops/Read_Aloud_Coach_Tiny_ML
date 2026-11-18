"""v2 merge: consolidate LS clip pool (v1 ls_clips.npz + v2 shards) and merge
with the Speech Commands caches into cacheV2_{split}.npz.
- kept LS words: >=40 clips in the full pool (same bar as v1; never padded)
- per-word cap 600, chosen by seeded shuffle of the pool
- speaker-disjoint split by LibriSpeech speaker id, same md5%10 rule as v1
"""
import os, sys, glob, hashlib
import numpy as np
import torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab import SC_COVERED

BASE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
MINN = 40
CAP = 600

# ---- consolidate pool ----
pool = {}  # word -> list of (wave, spk)
def add(waves, words, spks):
    for wv, w, s in zip(waves, words, spks):
        pool.setdefault(str(w), []).append((wv, str(s)))
d = np.load(os.path.join(BASE, "ls_clips.npz"), allow_pickle=True)
add(d["waves"], d["words"], d["spks"])
for pat in ["ls_shard_v2_*.npz", "ls_shard_tc_*.npz"]:
    for f in sorted(glob.glob(os.path.join(BASE, pat))):
        d = np.load(f, allow_pickle=True)
        add(d["waves"], d["words"], d["spks"])
        print("shard", os.path.basename(f), len(d["words"]), flush=True)
counts = {w: len(v) for w, v in sorted(pool.items())}
print("pool counts:", counts, flush=True)
np.savez_compressed(os.path.join(BASE, "ls_clips_v2.npz"),
                    waves=np.concatenate([np.stack([c for c, s in v]) for v in pool.values()]),
                    words=np.concatenate([np.array([w]*len(v)) for w, v in pool.items()]),
                    spks=np.concatenate([np.array([s for c, s in v]) for v in pool.values()]))
print("saved ls_clips_v2.npz", flush=True)

new_words = sorted([w for w, v in pool.items() if len(v) >= MINN])
print("kept LS words (>=40):", new_words, flush=True)

CLASSES = SC_COVERED + new_words + ["_other_", "_silence_"]
cidx = {c: i for i, c in enumerate(CLASSES)}
np.save(os.path.join(BASE, "classes_v2.npy"), np.array(CLASSES))
print("CLASSES", CLASSES, flush=True)

mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=512, hop_length=256, n_mels=40)
def to_mel(waves):
    out = []
    for i in range(0, len(waves), 512):
        m = mel_tf(torch.from_numpy(np.stack(waves[i:i+512])))
        out.append(torch.log(m + 1e-6).half().numpy())
    return np.concatenate(out)

rng = np.random.RandomState(123)
for split in ["train", "val", "test"]:
    d = np.load(os.path.join(BASE, f"cache_sc_{split}.npz"))
    sc_mel, sc_lab = d["mel"], d["label"]
    sc_classes = list(np.load(os.path.join(BASE, "classes_sc.npy")))
    remap = np.array([cidx[c] for c in sc_classes])
    mels = [sc_mel]; labs = [remap[sc_lab]]; srcs = [np.array(["speech_commands"]*len(sc_lab))]
    waves, labs_ls = [], []
    for w in new_words:
        items = list(pool[w])
        rng.shuffle(items)
        items = items[:CAP]
        for clip, spk in items:
            h = int(hashlib.md5(spk.encode()).hexdigest(), 16) % 10
            s = "train" if h <= 7 else ("val" if h == 8 else "test")
            if s == split:
                waves.append(clip); labs_ls.append(cidx[w])
    if waves:
        mels.append(to_mel(waves)); labs.append(np.array(labs_ls))
        srcs.append(np.array(["librispeech"]*len(labs_ls)))
    np.savez_compressed(os.path.join(BASE, f"cacheV2_{split}.npz"),
                        mel=np.concatenate(mels), label=np.concatenate(labs), source=np.concatenate(srcs))
    print("merged", split, np.concatenate(mels).shape, flush=True)
print("DONE", flush=True)

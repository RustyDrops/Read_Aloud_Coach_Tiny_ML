"""Prepare Speech Commands v0.02 mel cache (streaming, per-split, low memory).
Splits: official validation_list.txt / testing_list.txt (speaker-hash based).
Augmentation (train only, programmatic): pitch-up resample copies + background-noise mixes.
"""
import os, sys, glob, random
import numpy as np
import soundfile as sf
import torch, torchaudio
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab import SC_COVERED, SC_OTHER

random.seed(7); np.random.seed(7); torch.manual_seed(7)
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "sc_mine")
SR = 16000
assert os.path.isdir(DATA), DATA

def read_list(name):
    with open(os.path.join(DATA, name)) as f:
        return set(x.strip() for x in f if x.strip())

val_set = read_list("validation_list.txt")
test_set = read_list("testing_list.txt")
print("val list", len(val_set), "test list", len(test_set), flush=True)

mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=512, hop_length=256, n_mels=40)

def to_mel(waves):
    out = []
    for i in range(0, len(waves), 1000):
        chunk = torch.from_numpy(np.stack(waves[i:i+1000]))
        m = mel_tf(chunk)
        out.append(torch.log(m + 1e-6).half().numpy())
        del chunk, m
    return np.concatenate(out)

SKIPPED = [0]
def load_wav(path):
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception:
        SKIPPED[0] += 1; return None
    if sr != SR or len(x) == 0:
        SKIPPED[0] += 1; return None
    if len(x) > SR: x = x[:SR]
    elif len(x) < SR: x = np.pad(x, (0, SR-len(x)))
    return x.astype(np.float32)

noise_files = glob.glob(os.path.join(DATA, "_background_noise_", "*.wav"))
noises = [sf.read(f, dtype="float32")[0] for f in noise_files]
print("noise files", len(noises), flush=True)

def noise_slice(rng):
    n = noises[rng.randrange(len(noises))]
    s = rng.randrange(0, max(1, len(n)-SR))
    return n[s:s+SR].astype(np.float32)

CLASSES = SC_COVERED + ["_other_", "_silence_"]
cidx = {c: i for i, c in enumerate(CLASSES)}
print("CLASSES", CLASSES, flush=True)
CAP_TRAIN = {"word": 1000, "_other_": 2200, "_silence_": 900}
CAP_EVAL = {"word": 10000, "_other_": 500, "_silence_": 300}

def collect(word, split):
    files = sorted(glob.glob(os.path.join(DATA, word, "*.wav")))
    sel = []
    for f in files:
        rel = f"{word}/{os.path.basename(f)}"
        s = "test" if rel in test_set else ("val" if rel in val_set else "train")
        if s == split: sel.append(f)
    return sel

def process(split):
    rng = random.Random(11)
    waves, labels = [], []
    for w in SC_COVERED:
        files = collect(w, split)
        cap = CAP_TRAIN["word"] if split == "train" else CAP_EVAL["word"]
        rng.shuffle(files); files = files[:cap]
        for f in files:
            x = load_wav(f)
            if x is None: continue
            waves.append(x); labels.append(cidx[w])
    other_files = []
    for w in SC_OTHER:
        other_files += collect(w, split)
    cap = CAP_TRAIN["_other_"] if split == "train" else CAP_EVAL["_other_"]
    rng.shuffle(other_files); other_files = other_files[:cap]
    for f in other_files:
        x = load_wav(f)
        if x is None: continue
        waves.append(x); labels.append(cidx["_other_"])
    cap = CAP_TRAIN["_silence_"] if split == "train" else CAP_EVAL["_silence_"]
    for i in range(cap):
        waves.append(noise_slice(rng) * rng.choice([0.02, 0.05, 0.1, 0.3])); labels.append(cidx["_silence_"])
    labels = np.array(labels, dtype=np.int64)
    print(split, "clean waves", len(waves), flush=True)
    if split == "train":
        idxs = [i for i in range(len(labels)) if labels[i] < len(SC_COVERED)]
        rng.shuffle(idxs)
        aug_w, aug_l = [], []
        for i in idxs[:int(0.6*len(idxs))]:
            x = waves[i]
            factor = rng.choice([1.2, 1.3, 1.4])
            y = resample_poly(x, SR, int(SR*factor)).astype(np.float32)
            if len(y) < SR:
                st = rng.randrange(0, SR-len(y)+1); y = np.pad(y, (st, SR-len(y)-st))
            else: y = y[:SR]
            aug_w.append(y); aug_l.append(labels[i])
        for i in idxs[int(0.6*len(idxs)):int(0.9*len(idxs))]:
            x = waves[i].copy(); n = noise_slice(rng)
            snr = rng.uniform(5, 20)
            px = np.mean(x**2) + 1e-9; pn = np.mean(n**2) + 1e-9
            aug_w.append((x + n * np.sqrt(px / pn / (10**(snr/10)))).astype(np.float32)); aug_l.append(labels[i])
        waves = waves + aug_w; labels = np.concatenate([labels, np.array(aug_l)])
        print("train augmented ->", len(waves), flush=True)
    m = to_mel(waves)
    np.savez_compressed(os.path.join(BASE, f"cache_sc_{split}.npz"), mel=m, label=labels)
    print("saved", split, m.shape, "skipped so far", SKIPPED[0], flush=True)
    del waves, m

for split in ["train", "val", "test"]:
    process(split)
np.save(os.path.join(BASE, "classes_sc.npy"), np.array(CLASSES))
print("DONE", flush=True)

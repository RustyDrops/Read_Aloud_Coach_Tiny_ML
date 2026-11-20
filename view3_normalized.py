"""Step 3 (PLAN-VIEWS-SPECIALISTS): View 3 solo test — child-normalized mel.
Zero training: warp JIBO DEV mels by inverse VTLP (alpha < 1, same warp
machinery as build_cache_v6) to map child speech toward the adult register
v2 was trained on, then score the FROZEN v2 int8 model unchanged.
DEV ONLY — the spent C4/G1 child exam is never touched.
Also: (i) caches per-clip F0 features for dev (child_eval/dev_f0.npz) for
the S1 trigger, (ii) audits the S1 register gate (median F0 >= 220 Hz) on
adult LibriSpeech pool waves from val/test speakers (fire rate ~0 expected).
Outputs: metrics_view3.json.
"""
import os, json, glob, hashlib
import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.abspath(__file__))
CE = os.path.join(BASE, "child_eval")
SR = 16000
VOCAB = ["down","go","one","three","two","cat","dog","bird","tree","house","bed","happy","stop",
         "yes","no","on","a","and","away","can","come","for","here","in","is","little","look",
         "make","me","my","not","said","see","the","to","up","we","where","you"]
# NOTE: keep class order identical to phase0/metrics_v2 — load from metrics file instead
import json as _json
CLASSES = _json.load(open(os.path.join(BASE, "metrics_v2.json")))["classes"]
CIDX = {c: i for i, c in enumerate(CLASSES)}
TAU = 0.30

import torch, torchaudio
mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=512, hop_length=256, n_mels=40)
metrics = json.load(open(os.path.join(BASE, "metrics_v2.json")))
MU, SD = metrics["mel_mean"], metrics["mel_std"]

import sys
sys.path.insert(0, BASE)
from build_cache_v6 import warp_matrix  # same piecewise-linear VTLP machinery

def load_segment(rel, start, end):
    path = os.path.join(CE, "jibo_release", "jibo_release", "data", *rel.split("/"))
    info = sf.info(path); sr = info.samplerate
    a = int(round(start * sr)); b = int(round(end * sr))
    x, _ = sf.read(path, start=max(0, a), frames=max(0, b - a), dtype="float32", always_2d=False)
    if x.ndim > 1: x = x[:, 0]
    if sr != SR:
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    seg = x
    if len(seg) == 0: return None
    if len(seg) > SR:
        e = np.concatenate([[0.0], np.cumsum(seg.astype(np.float64) ** 2)])
        win = e[SR:] - e[:-SR]
        s0 = int(np.argmax(win)); seg = seg[s0:s0 + SR]
    elif len(seg) < SR:
        seg = np.pad(seg, (0, SR - len(seg)))
    return seg.astype(np.float32)

def f0_feats(w):
    fr_len, hop = 400, 160
    f0s = []
    for s in range(0, len(w) - fr_len, hop):
        fr = w[s:s + fr_len]
        if np.sqrt((fr ** 2).mean()) < 0.01: continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[fr_len - 1:]
        if ac[0] <= 0: continue
        lo, hi = int(SR / 600), int(SR / 80)
        seg = ac[lo:hi]
        if len(seg) == 0: continue
        lag = lo + int(np.argmax(seg))
        if seg.max() / ac[0] > 0.3: f0s.append(SR / lag)
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    freqs = np.fft.rfftfreq(len(w), 1 / SR)
    cent = float((freqs * spec).sum() / max(spec.sum(), 1e-9))
    env = np.abs(w)
    dur = float((env > 0.1 * env.max()).mean()) if env.max() > 0 else 0.0
    return (float(np.median(f0s)) if f0s else 0.0,
            float(np.percentile(f0s, 75) - np.percentile(f0s, 25)) if len(f0s) > 3 else 0.0,
            float(len(f0s) * hop / SR), cent, dur)

# ---------- dev waves + mels + F0 ----------
dev = json.load(open(os.path.join(CE, "dev_manifest.json")))["clips"]
waves, keep = [], []
for c in dev:
    w = load_segment(c["audio_rel"], c["start_s"], c["end_s"])
    if w is None: continue
    waves.append(w); keep.append(c)
W = np.stack(waves)
print("dev waves", W.shape, flush=True)
F = np.array([f0_feats(w) for w in W])
np.savez(os.path.join(CE, "dev_f0.npz"), F=F,
         clip_id=np.array([c["clip_id"] for c in keep]),
         child_id=np.array([c["child_id"] for c in keep]),
         word=np.array([c["word"] for c in keep]))
mels = []
for i in range(0, len(W), 256):
    m = mel_tf(torch.from_numpy(W[i:i + 256]))
    mels.append(torch.log(m + 1e-6).numpy())
X = np.concatenate(mels).astype(np.float32)
y = np.array([CIDX[c["word"]] for c in keep])
np.savez(os.path.join(CE, "dev_mel.npz"), mel=X.astype(np.float16), y=y,
         word=np.array([c["word"] for c in keep]),
         child_id=np.array([c["child_id"] for c in keep]))

import onnxruntime as ort
sess = ort.InferenceSession(os.path.join(BASE, "model_v2_int8.onnx"), providers=["CPUExecutionProvider"])
def probs(melarr):
    outs = []
    for i in range(0, len(melarr), 512):
        xb = ((melarr[i:i + 512] - MU) / SD).astype(np.float32)[:, None, :, :]
        outs.append(sess.run(None, {"mel": xb})[0])
    lg = np.concatenate(outs); lg = lg - lg.max(axis=1, keepdims=True)
    P = np.exp(lg); P /= P.sum(axis=1, keepdims=True)
    return P

def score(P, tag):
    pred = P.argmax(1)
    accept_true = (pred == y) & (P[np.arange(len(y)), y] >= TAU)
    return {"tag": tag, "argmax_acc": round(float((pred == y).mean()), 4),
            "true_accept": round(float(accept_true.mean()), 4)}

R = {"n_dev": int(len(y)), "alphas": {}}
P0 = probs(X)
R["alphas"]["1.00_unwarped"] = score(P0, "unwarped")
print(R["alphas"]["1.00_unwarped"], flush=True)
for alpha in [0.87, 0.80, 0.74]:
    Wm = warp_matrix(alpha)
    Xw = np.einsum("oi,bit->bot", Wm, X).astype(np.float32)
    Pw = probs(Xw)
    R["alphas"][f"{alpha:.2f}"] = score(Pw, f"alpha={alpha}")
    print(R["alphas"][f"{alpha:.2f}"], flush=True)
base = R["alphas"]["1.00_unwarped"]
for k, v in R["alphas"].items():
    v["delta_argmax_vs_unwarped"] = round(v["argmax_acc"] - base["argmax_acc"], 4)
    v["delta_true_accept_vs_unwarped"] = round(v["true_accept"] - base["true_accept"], 4)
R["gate_fire_rate_dev_f0_ge_220"] = round(float((F[:, 0] >= 220).mean()), 4)

# ---------- adult gate audit: LS pool waves, val/test speakers ----------
pool_waves, pool_spks = [], []
d = np.load(os.path.join(BASE, "ls_clips.npz"), allow_pickle=True)
pool_waves.append(d["waves"]); pool_spks.append(d["spks"])
for pat in ["ls_shard_v2_*.npz", "ls_shard_tc_*.npz"]:
    for f in sorted(glob.glob(os.path.join(BASE, pat))):
        dd = np.load(f, allow_pickle=True)
        pool_waves.append(dd["waves"]); pool_spks.append(dd["spks"])
PW = np.concatenate(pool_waves); PS = np.concatenate(pool_spks)
h = np.array([int(hashlib.md5(s.encode()).hexdigest(), 16) % 10 for s in PS])
adult_mask = h >= 8  # val (8) + test (9) speakers — never in v2 train
idx = np.where(adult_mask)[0]
rng = np.random.RandomState(7)
if len(idx) > 2000: idx = rng.choice(idx, 2000, replace=False)
Fa = np.array([f0_feats(PW[i]) for i in idx])
R["adult_gate_audit"] = {
    "n_ls_val_test_speaker_clips": int(len(idx)),
    "f0_median_hz_p50": round(float(np.median(Fa[:, 0])), 1),
    "f0_median_hz_p95": round(float(np.percentile(Fa[:, 0], 95)), 1),
    "fire_rate_f0_ge_220": round(float((Fa[:, 0] >= 220).mean()), 4),
    "fire_rate_f0_ge_200": round(float((Fa[:, 0] >= 200).mean()), 4),
    "note": "LibriSpeech adult readers, val/test speakers only. SC speakers are adults/teens of similar register; JIBO dev fire rate is the child-side counterpart."}
print(R["adult_gate_audit"], flush=True)

with open(os.path.join(BASE, "metrics_view3.json"), "w") as f:
    json.dump(R, f, indent=1)
print(json.dumps(R, indent=1), flush=True)
print("VIEW3_DONE", flush=True)

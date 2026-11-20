"""Phase 0 diagnostics for PLAN-VIEWS-SPECIALISTS (Read-Aloud).
Local CPU only. DEV set ONLY — the C4/G1 child EXAM is spent and is
never touched here. No training of any candidate model; the only fitted
objects are tiny diagnostic probes on dev, split child-disjoint, and
they are reported as diagnostics, not gates.
Outputs: phase0_results.json + phase0_dev_probs.npz (local only).
"""
import os, json, hashlib
import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.abspath(__file__))
RA = os.path.dirname(BASE)
SR = 16000
VOCAB = ["down","go","one","three","two","cat","dog","bird","tree","house","bed","happy","stop",
         "yes","no","on","a","and","away","can","come","for","here","in","is","it","little","look",
         "make","me","my","not","said","see","the","to","up","we","where","you"]
CLASSES = VOCAB + ["_other_", "_silence_"]
CIDX = {c: i for i, c in enumerate(CLASSES)}
TAU = 0.30

import torch, torchaudio
mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=512, hop_length=256, n_mels=40)
metrics = json.load(open(os.path.join(RA, "metrics_v2.json")))
MU, SD = metrics["mel_mean"], metrics["mel_std"]

def load_segment(rel, start, end):
    path = os.path.join(BASE, "jibo_release", "jibo_release", "data", *rel.split("/"))
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

dev = json.load(open(os.path.join(BASE, "dev_manifest.json")))["clips"]
waves, keep = [], []
for c in dev:
    w = load_segment(c["audio_rel"], c["start_s"], c["end_s"])
    if w is None: continue
    waves.append(w); keep.append(c)
W = np.stack(waves)
print("dev waves", W.shape, flush=True)
mels = []
for i in range(0, len(W), 256):
    m = mel_tf(torch.from_numpy(W[i:i + 256]))
    mels.append(torch.log(m + 1e-6).numpy())
X = np.concatenate(mels).astype(np.float32)

import onnxruntime as ort
sess = ort.InferenceSession(os.path.join(RA, "model_v2_int8.onnx"), providers=["CPUExecutionProvider"])
def probs(melarr, mu, sd):
    outs = []
    for i in range(0, len(melarr), 512):
        xb = ((melarr[i:i + 512] - mu) / sd).astype(np.float32)[:, None, :, :]
        outs.append(sess.run(None, {"mel": xb})[0])
    lg = np.concatenate(outs); lg = lg - lg.max(axis=1, keepdims=True)
    P = np.exp(lg); P /= P.sum(axis=1, keepdims=True)
    return P

P = probs(X, MU, SD)
y = np.array([CIDX[c["word"]] for c in keep])
pred = P.argmax(1)
order = np.argsort(-P, axis=1)
rank_true = np.array([int(np.where(order[i] == y[i])[0][0]) + 1 for i in range(len(y))])
p_true = P[np.arange(len(y)), y]
accept_true = (pred == y) & (p_true >= TAU)  # shipped rule exactly: p>=tau AND p - max_other >= 0 (margin 0) <=> argmax is true and p>=tau. Reproduces C4's 0.3894.
np.savez(os.path.join(BASE, "phase0_dev_probs.npz"), P=P, y=y,
         clip_id=np.array([c["clip_id"] for c in keep]),
         child_id=np.array([c["child_id"] for c in keep]),
         word=np.array([c["word"] for c in keep]),
         grade=np.array([c["grade"] for c in keep]))
R = {}
n = len(y)
R["n_dev"] = n
R["argmax_acc"] = round(float((pred == y).mean()), 4)
R["true_accept"] = round(float(accept_true.mean()), 4)

# ---------- (a) false-rejection rescue headroom ----------
FR = ~accept_true
R["false_rejections"] = int(FR.sum())
fr = {}
fr["pred_is_true_but_quiet"] = int(((pred == y) & FR).sum())
fr["pred_is_other"] = int(((pred == CIDX["_other_"]) & FR).sum())
fr["pred_is_silence"] = int(((pred == CIDX["_silence_"]) & FR).sum())
fr["pred_is_wrong_word"] = int(FR.sum() - fr["pred_is_true_but_quiet"] - fr["pred_is_other"] - fr["pred_is_silence"])
rt = rank_true[FR]
fr["rank_of_true_word"] = {f"rank_{k}": int((rt == k).sum()) for k in [1, 2, 3, 4, 5]}
fr["rank_gt5"] = int((rt > 5).sum())
fr["p_true_quantiles"] = {q: round(float(np.quantile(p_true[FR], v)), 4) for q, v in
                          [("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9)]}
def oracle_ta(mask_fr):
    """true-accept if a perfect rescue accepted FR clips matching mask (over FR indices)."""
    idx = np.where(FR)[0][mask_fr]
    acc = accept_true.copy(); acc[idx] = True
    return round(float(acc.mean()), 4)
fr["oracle_true_accept_rescue_rank_le2"] = oracle_ta(rt <= 2)
fr["oracle_true_accept_rescue_rank_le3"] = oracle_ta(rt <= 3)
fr["oracle_true_accept_rescue_pred_true_only"] = oracle_ta((pred[FR] == y[FR]))
# argmax-side oracle: perfect re-ranker promoting true word when it is rank 2
r2 = FR & (rank_true == 2)
fr["oracle_argmax_promote_rank2_count"] = int(r2.sum())
fr["oracle_argmax_if_rank2_promoted"] = round(float(((pred == y) | r2).mean()), 4)
# FR by word
fr["by_word"] = {}
for w in sorted(set(c["word"] for c in keep)):
    m = np.array([c["word"] == w for c in keep])
    fr["by_word"][w] = {"n": int(m.sum()), "fr": int((FR & m).sum()),
                        "fr_pred_other": int(((pred == CIDX["_other_"]) & FR & m).sum()),
                        "fr_rank_le2": int((FR & m & (rank_true <= 2)).sum()),
                        "fr_rank_le3": int((FR & m & (rank_true <= 3)).sum())}
R["false_rejection_rescue"] = fr

# ---------- (b) pair-verifier headroom ----------
def pair_stats(tword, pword):
    m = np.array([c["word"] == tword for c in keep])
    idx = np.where(m)[0]
    ti, pi = CIDX[tword], CIDX[pword]
    top2 = order[idx][:, :2]
    return {
        "n_true": int(m.sum()),
        "pred_is_pair": int((pred[idx] == pi).sum()),
        "true_rank_le2": int((rank_true[idx] <= 2).sum()),
        "top2_is_exactly_pair": int((np.array([set(r) == {ti, pi} for r in top2])).sum()),
        "pair_in_top2": int((np.array([pi in r for r in top2])).sum()),
        "oracle_gain_clips": int(((pred[idx] == pi) & (rank_true[idx] <= 2)).sum()),
    }
pairs = {}
pairs["a_on_from_a"] = pair_stats("a", "on")
pairs["on_a_from_on"] = pair_stats("on", "a")
pairs["three_tree_from_three"] = pair_stats("three", "tree")
pairs["tree_three_from_tree"] = pair_stats("tree", "three")
R["pair_headroom"] = pairs

# ---------- (c) 'a' diagnosis ----------
ma = np.array([c["word"] == "a" for c in keep])
from collections import Counter
conf_a = Counter(CLASSES[p] for p in pred[ma])
R["a_confusion_row"] = dict(conf_a.most_common())
R["a_rank"] = {"rank_le2": int((rank_true[ma] <= 2).sum()), "rank_le3": int((rank_true[ma] <= 3).sum()),
               "n": int(ma.sum()), "p_true_median": round(float(np.median(p_true[ma])), 4)}

# F0 + envelope features per clip (numpy autocorrelation, child range 80-600 Hz)
def f0_feats(w):
    fr_len, hop = 400, 160
    f0s, cents = [], []
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

F = np.array([f0_feats(w) for w in W])  # f0_med, f0_iqr, voiced_s, centroid, dur_frac
R["f0_by_word"] = {}
for w_ in sorted(set(c["word"] for c in keep)):
    m = np.array([c["word"] == w_ for c in keep])
    R["f0_by_word"][w_] = {"n": int(m.sum()), "f0_median_hz": round(float(np.median(F[m, 0])), 1),
                           "voiced_s_median": round(float(np.median(F[m, 2])), 3)}

# child-disjoint probe: pitch/formant features vs mel-mean features, task a-vs-rest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
mel_mean = X.mean(axis=2)  # (n,40)
formant = np.stack([X[:, b0:b1, :].mean(axis=(1, 2)) for b0, b1 in [(0, 5), (5, 10), (10, 16), (16, 24), (24, 32), (32, 40)]], axis=1)
PF = np.concatenate([F, formant], axis=1)
child = np.array([c["child_id"] for c in keep])
par = np.array([int(hashlib.md5(ch.encode()).hexdigest(), 16) % 2 for ch in child])
is_a = (y == CIDX["a"]).astype(int)
def probe(feats, name):
    out = {}
    for test_par in [0, 1]:
        tr, te = par != test_par, par == test_par
        if is_a[tr].sum() == 0 or is_a[te].sum() == 0: continue
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit((feats[tr] - mu) / sd, is_a[tr])
        p = clf.predict_proba((feats[te] - mu) / sd)[:, 1]
        out[f"fold{test_par}"] = {"auc": round(float(roc_auc_score(is_a[te], p)), 3),
                                  "n_test": int(te.sum()), "n_a_test": int(is_a[te].sum())}
    out["mean_auc"] = round(float(np.mean([v["auc"] for v in out.values() if "auc" in v])), 3) if out else None
    return out
R["probe_a_vs_rest"] = {"pitch_formant_duration": probe(PF, "pf"),
                        "mel_mean_only": probe(mel_mean, "mel"),
                        "combined": probe(np.concatenate([PF, mel_mean], axis=1), "both")}
R["probe_note"] = "Diagnostic only: fitted on dev, child-disjoint folds within dev; NOT a gate, NOT a candidate."

# ---------- adult diagnostics (v2 caches are gone; use v6 adult probe + v3 test mapped) ----------
adult = {}
try:
    d = np.load(os.path.join(RA, "cacheV6_adultprobe.npz"))
    Pa = probs(d["mel"].astype(np.float32), MU, SD)
    ya = d["label"]; preda = Pa.argmax(1)
    ar = np.array([int(np.where(np.argsort(-Pa[i]) == ya[i])[0][0]) + 1 for i in range(len(ya))])
    pta = Pa[np.arange(len(ya)), ya]
    adult["v6_vtlp_warped_adult_probe"] = {
        "n": int(len(ya)), "argmax_acc": round(float((preda == ya).mean()), 4),
        "note": "alpha=1.25 VTLP-warped adults built for v6; G1 recorded v2=0.826 here",
        "false_rejections": int((pta < TAU).sum()),
        "fr_rank_le2_frac_of_fr": round(float(((ar <= 2) & (pta < TAU)).sum() / max((pta < TAU).sum(), 1)), 4)}
except Exception as e:
    adult["v6_vtlp_warped_adult_probe"] = {"error": str(e)}
try:
    c3 = list(np.load(os.path.join(RA, "classes_v3.npy"), allow_pickle=True))
    d = np.load(os.path.join(RA, "cacheV3_test.npz"))
    keep3 = np.array([c3[l] in CIDX for l in d["label"]])
    y3 = np.array([CIDX[c3[l]] for l in d["label"][keep3]])
    # v3 mel normalisation unknown; use v2 MU/SD (same 40x63 pipeline family) and label as diagnostic
    P3 = probs(d["mel"][keep3].astype(np.float32), MU, SD)
    pred3 = P3.argmax(1); pt3 = P3[np.arange(len(y3)), y3]
    adult["v3_test_mapped_to_v2_classes"] = {
        "n": int(len(y3)), "argmax_acc_v2_on_it": round(float((pred3 == y3).mean()), 4),
        "false_rejections": int((pt3 < TAU).sum()),
        "note": "DIAGNOSTIC ONLY: v3's adult test clips relabelled by class name; v3 mel normalisation constants not reused, v2's applied; not an exam."}
except Exception as e:
    adult["v3_test_mapped_to_v2_classes"] = {"error": str(e)}
R["adult_diagnostics"] = adult
R["adult_reference_from_metrics_v2"] = {"val_acc": metrics["val_best_acc"],
    "val_true_accept": metrics["decision_rule"]["val_true_accept"],
    "val_false_accept": metrics["decision_rule"]["val_false_accept"],
    "a_adult_test": metrics["per_word_test"]["a"]}

with open(os.path.join(BASE, "phase0_results.json"), "w") as f:
    json.dump(R, f, indent=1)
print(json.dumps(R, indent=1), flush=True)
print("PHASE0_DONE", flush=True)

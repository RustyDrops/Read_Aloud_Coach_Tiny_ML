"""Step 1 (PLAN-VIEWS-SPECIALISTS): build S1 (child-register rescue verifier)
and S2 (three/tree pair verifier). Local CPU, 0 Colab units.

Training data: cacheV6_train.npz ONLY — the two license-logged v2 corpora
(Speech Commands + LibriSpeech-mined) as originals + VTLP-warped copies.
Child corpora are used for EVALUATION ONLY (JIBO dev), never trained on.
The spent C4/G1 child exam is not touched anywhere in this script.

S1: logistic verifier on (clip, target) pairs. Pair population = target in
v2's top-2 word classes (the inference trigger population). Features:
v2 prob features + mel band/temporal stats + target one-hot.
Gate at inference: v2 rejects target AND dev/adult F0 median >= 220 Hz
(child register) AND rank(target) <= 2. Threshold frozen on a group-disjoint
holdout of the v6 train cache (originals and their warped copies stay in
the same group) BEFORE any dev scoring.

S2: logistic three-vs-tree on clips whose v2 top-2 is exactly {three, tree}
(system use) trained on all three/tree clips in the v6 train cache.

Evals: JIBO dev (development numbers, labelled), adult val (rebuilt
cacheV2_val, verified against metrics_v2), warped-adult probe.
Outputs: metrics_s1.json, metrics_s2.json, s1_model.npz, s2_model.npz.
"""
import os, json, hashlib
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CE = os.path.join(BASE, "child_eval")
TAU = 0.30
CLASSES = json.load(open(os.path.join(BASE, "metrics_v2.json")))["classes"]
CIDX = {c: i for i, c in enumerate(CLASSES)}
NC = len(CLASSES)
WORD_IDX = np.array([i for i, c in enumerate(CLASSES) if not c.startswith("_")])
MU = json.load(open(os.path.join(BASE, "metrics_v2.json")))["mel_mean"]
SD = json.load(open(os.path.join(BASE, "metrics_v2.json")))["mel_std"]

import onnxruntime as ort
sess = ort.InferenceSession(os.path.join(BASE, "model_v2_int8.onnx"), providers=["CPUExecutionProvider"])
def v2_probs(melarr):
    outs = []
    for i in range(0, len(melarr), 512):
        xb = ((melarr[i:i + 512].astype(np.float32) - MU) / SD)[:, None, :, :]
        outs.append(sess.run(None, {"mel": xb})[0])
    lg = np.concatenate(outs); lg = lg - lg.max(axis=1, keepdims=True)
    P = np.exp(lg); P /= P.sum(axis=1, keepdims=True)
    return P

def pair_features(mel, P, e):
    """mel (40,63) log-mel, P (42,) v2 probs, e target class idx."""
    p = P
    order = np.argsort(-p)
    top1, top2 = order[0], order[1]
    rank_e = int(np.where(order == e)[0][0]) + 1
    ent = float(-(p * np.log(p + 1e-9)).sum())
    band_mean = mel.mean(axis=1); band_std = mel.std(axis=1)
    t3 = np.array([mel[:, :21].mean(), mel[:, 21:42].mean(), mel[:, 42:].mean()])
    onehot = np.zeros(len(WORD_IDX), dtype=np.float32); onehot[list(WORD_IDX).index(e)] = 1.0
    scal = np.array([p[e], p[top1], p[top2], p[e] - p[top1], min(rank_e, 5) / 5.0,
                     ent, p[CIDX["_other_"]], p[CIDX["_silence_"]], 1.0 if top1 == e else 0.0],
                    dtype=np.float32)
    return np.concatenate([scal, band_mean, band_std, t3, onehot]).astype(np.float32)

def accept_matrix(P):
    top = np.argsort(P, axis=1); t1, t2 = top[:, -1], top[:, -2]
    p1 = P[np.arange(len(P)), t1]; p2 = P[np.arange(len(P)), t2]
    maxother = np.where(np.arange(NC)[None, :] == t1[:, None], p2[:, None], p1[:, None])
    return (P >= TAU) & ((P - maxother) >= 0.0)

def system_stats(P, y, extra_accept):
    """extra_accept: (N, NC) bool of verifier-added accepts."""
    acc = accept_matrix(P) | extra_accept
    not_true = np.ones_like(acc); not_true[np.arange(len(y)), y] = False
    fa = float(acc[not_true & np.isin(np.arange(NC), WORD_IDX)[None, :]].mean())
    mw = np.isin(y, WORD_IDX)
    ta = float(acc[mw, y[mw]].mean())
    base = accept_matrix(P)
    fa0 = float(base[not_true & np.isin(np.arange(NC), WORD_IDX)[None, :]].mean())
    ta0 = float(base[mw, y[mw]].mean())
    return {"true_accept": round(ta, 4), "false_accept_all_pairs": round(fa, 4),
            "v2_true_accept": round(ta0, 4), "v2_false_accept_all_pairs": round(fa0, 4),
            "delta_true_accept": round(ta - ta0, 4), "delta_false_accept": round(fa - fa0, 4)}

# ================= load v6 train cache + v2 probs =================
d6 = np.load(os.path.join(BASE, "cacheV6_train.npz"))
MEL6 = d6["mel"].astype(np.float32); Y6 = d6["label"]; WARP6 = d6["warped"]; SRC6 = d6["source"]
N_ORIG = int((~WARP6).sum())
nonsil_orig = np.where(Y6[:N_ORIG] != CIDX["_silence_"])[0]
group = np.arange(len(Y6))
group[N_ORIG:] = nonsil_orig  # warped copy -> its original's group id
print(f"v6 train {len(Y6)} orig {N_ORIG}", flush=True)
P6 = v2_probs(MEL6)
np.savez(os.path.join(BASE, "v6train_probs.npz"), P=P6)
print("v2 probs on v6 train done", flush=True)

holdout = np.array([int(hashlib.md5(str(int(g)).encode()).hexdigest(), 16) % 10 == 0 for g in group])

# ================= S1 dataset =================
def build_pairs(idxs):
    X, yy, meta = [], [], []
    for i in idxs:
        top2 = np.argsort(-P6[i])[:2]
        for e in top2:
            if e not in WORD_IDX: continue
            X.append(pair_features(MEL6[i], P6[i], int(e)))
            yy.append(1 if int(e) == int(Y6[i]) else 0)
            meta.append((i, int(e)))
    return np.stack(X), np.array(yy), meta

tr_idx = np.where(~holdout)[0]; ho_idx = np.where(holdout)[0]
Xtr, ytr, _ = build_pairs(tr_idx)
Xho, yho, ho_meta = build_pairs(ho_idx)
print(f"S1 pairs: train {len(ytr)} (pos {ytr.sum()}), holdout {len(yho)} (pos {yho.sum()})", flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler().fit(Xtr)
s1 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
s1.fit(sc1.transform(Xtr), ytr)
pho = s1.predict_proba(sc1.transform(Xho))[:, 1]

# freeze threshold on holdout TRIGGER population: v2 rejects e, rank(e)<=2 (by construction), clip warped or not
trig = []
for k, (i, e) in enumerate(ho_meta):
    base_acc = P6[i, e] >= TAU and np.argmax(P6[i]) == e
    trig.append(not base_acc)
trig = np.array(trig)
best_thr, best_key = None, None
for thr in np.arange(0.30, 0.951, 0.025):
    pred = (pho >= thr) & trig
    npos = (yho == 1) & trig; nneg = (yho == 0) & trig
    rescue = float(pred[npos].mean()) if npos.sum() else 0.0
    fa = float(pred[nneg].mean()) if nneg.sum() else 0.0
    if fa <= 0.02:
        key = (rescue, -thr)
        if best_key is None or key > best_key: best_key, best_thr = (key, float(thr))
if best_thr is None:  # fall back to strictest
    best_thr = 0.95
rescue_ho = float(((pho >= best_thr) & trig & (yho == 1)).sum() / max(((yho == 1) & trig).sum(), 1))
fa_ho = float(((pho >= best_thr) & trig & (yho == 0)).sum() / max(((yho == 0) & trig).sum(), 1))
from sklearn.metrics import roc_auc_score
S1 = {"threshold_frozen_on_holdout": best_thr,
      "holdout_trigger_pairs": int(trig.sum()),
      "holdout_rescue_rate_of_triggered_positives": round(rescue_ho, 4),
      "holdout_false_accept_rate_of_triggered_negatives": round(fa_ho, 4),
      "holdout_auc_all_pairs": round(float(roc_auc_score(yho, pho)), 4),
      "n_train_pairs": int(len(ytr)), "n_holdout_pairs": int(len(yho)),
      "feature_dim": int(Xtr.shape[1])}
print("S1", S1, flush=True)
np.savez(os.path.join(BASE, "s1_model.npz"), coef=s1.coef_[0], intercept=s1.intercept_,
         scaler_mean=sc1.mean_, scaler_scale=sc1.scale_, threshold=np.array([best_thr]),
         classes=np.array(CLASSES))

def s1_verifier_probs(melarr, Parr, targets):
    """targets: (N,) target class idx per row (rows may repeat clips). Returns verifier prob per row."""
    X = np.stack([pair_features(melarr[i], Parr[i], int(targets[i])) for i in range(len(targets))])
    return s1.predict_proba(sc1.transform(X))[:, 1]

def rescue_eval(P, y, melarr, f0med=None, tag=""):
    """Apply S1 system rule. f0med None -> ungated (worst case)."""
    N = len(y)
    base = accept_matrix(P)
    order = np.argsort(-P, axis=1)
    rank = np.empty_like(order)
    for i in range(N): rank[i, order[i]] = np.arange(1, NC + 1)
    extra = np.zeros_like(base)
    # candidate pairs: target in top-2 word classes, not base-accepted
    cand = []
    for i in range(N):
        for e in order[i, :2]:
            if e in WORD_IDX and not base[i, e]:
                if f0med is not None and f0med[i] < 220: continue
                cand.append((i, int(e)))
    if cand:
        X = np.stack([pair_features(melarr[i], P[i], e) for i, e in cand])
        vp = s1.predict_proba(sc1.transform(X))[:, 1]
        for k, (i, e) in enumerate(cand):
            if vp[k] >= best_thr: extra[i, e] = True
    st = system_stats(P, y, extra)
    # verifier-corrected argmax: true word rescued counts as corrected top-1
    pred = P.argmax(1)
    corr = (pred == y) | extra[np.arange(N), y]
    st["argmax_acc"] = round(float((pred == y).mean()), 4)
    st["verifier_corrected_argmax"] = round(float(corr.mean()), 4)
    st["n_rescued_true"] = int((extra[np.arange(N), y] & ~base[np.arange(N), y]).sum())
    st["tag"] = tag
    return st, extra

# ================= eval: child dev =================
dp = np.load(os.path.join(CE, "phase0_dev_probs.npz"), allow_pickle=True)
dm = np.load(os.path.join(CE, "dev_mel.npz"), allow_pickle=True)
df0 = np.load(os.path.join(CE, "dev_f0.npz"), allow_pickle=True)
assert (dp["clip_id"] == df0["clip_id"]).all() and (dp["word"] == dm["word"]).all() and (dp["y"] == dm["y"]).all()
Pd, yd = dp["P"], dp["y"]
meld = dm["mel"].astype(np.float32)
f0d = df0["F"][:, 0]
dev_gated, extra_dev = rescue_eval(Pd, yd, meld, f0med=f0d, tag="child_dev_gated_f0>=220")
dev_ungated, _ = rescue_eval(Pd, yd, meld, f0med=None, tag="child_dev_ungated")
S1["child_dev_DEVELOPMENT_numbers"] = {"gated": dev_gated, "ungated": dev_ungated}
print("S1 dev gated:", dev_gated, flush=True)
print("S1 dev ungated:", dev_ungated, flush=True)

# per-word true-accept on dev, gated system
acc_dev = accept_matrix(Pd) | extra_dev
S1["child_dev_per_word_true_accept_gated"] = {}
for w in sorted(set(dp["word"].tolist())):
    m = dp["word"] == w
    S1["child_dev_per_word_true_accept_gated"][w] = {
        "n": int(m.sum()), "v2": round(float(accept_matrix(Pd)[m, yd[m]].mean()), 4),
        "s1": round(float(acc_dev[m, yd[m]].mean()), 4)}

# ================= eval: adult val =================
av = np.load(os.path.join(BASE, "adult_val_probs.npz"), allow_pickle=True)
dv = np.load(os.path.join(BASE, "cacheV2_val.npz"))
Pav, yav = av["P"], av["y"]; melav = dv["mel"].astype(np.float32)
adult_ungated, _ = rescue_eval(Pav, yav, melav, f0med=None, tag="adult_val_ungated_worst_case")
S1["adult_val"] = {"ungated_worst_case": adult_ungated,
                   "note": "Val cache has no waveforms, so the F0 gate cannot be applied row-wise; "
                           "ungated is the absolute worst case. Measured adult LS gate fire rate at "
                           "220 Hz is in metrics_view3.json (adult_gate_audit). Gated estimate below "
                           "scales the added accepts by that fire rate.",
                   "fire_rate_adjusted_estimate": None}
adj = dict(adult_ungated)
fr = json.load(open(os.path.join(BASE, "metrics_view3.json")))["adult_gate_audit"]["fire_rate_f0_ge_220"]
adj["delta_true_accept"] = round(adult_ungated["delta_true_accept"] * fr, 4)
adj["delta_false_accept"] = round(adult_ungated["delta_false_accept"] * fr, 4)
adj["true_accept"] = round(adult_ungated["v2_true_accept"] + adj["delta_true_accept"], 4)
adj["false_accept_all_pairs"] = round(adult_ungated["v2_false_accept_all_pairs"] + adj["delta_false_accept"], 4)
S1["adult_val"]["fire_rate_adjusted_estimate"] = adj
print("S1 adult ungated:", adult_ungated, flush=True)

# ================= eval: warped-adult probe =================
pr = np.load(os.path.join(BASE, "cacheV6_adultprobe.npz"))
Ppr = v2_probs(pr["mel"].astype(np.float32)); ypr = pr["label"]
probe_ungated, _ = rescue_eval(Ppr, ypr, pr["mel"].astype(np.float32), f0med=None, tag="warped_adult_probe_ungated")
S1["warped_adult_probe"] = probe_ungated
print("S1 probe:", probe_ungated, flush=True)

with open(os.path.join(BASE, "metrics_s1.json"), "w") as f:
    json.dump(S1, f, indent=1)
print("S1_DONE", flush=True)

# ================= S2 three/tree =================
i3, it = CIDX["three"], CIDX["tree"]
tr_mask = np.isin(Y6, [i3, it])
X2, y2 = [], []
for i in np.where(tr_mask)[0]:
    mel = MEL6[i]
    band_mean = mel.mean(axis=1); band_std = mel.std(axis=1)
    t3 = np.array([mel[:, :21].mean(), mel[:, 21:42].mean(), mel[:, 42:].mean()])
    # fricative-head proxy: high-band energy share in first third vs rest
    hi = mel[24:, :].mean(axis=0)
    fr = np.array([hi[:21].mean() - hi[21:].mean(), mel[:, :10].mean(), mel[:, -10:].mean()])
    X2.append(np.concatenate([band_mean, band_std, t3, fr,
                              [P6[i, i3], P6[i, it], P6[i, i3] - P6[i, it]]]).astype(np.float32))
    y2.append(1 if Y6[i] == i3 else 0)
X2 = np.stack(X2); y2 = np.array(y2)
g2 = group[tr_mask]
ho2 = np.array([int(hashlib.md5(str(int(g)).encode()).hexdigest(), 16) % 10 == 0 for g in g2])
sc2 = StandardScaler().fit(X2[~ho2])
s2 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
s2.fit(sc2.transform(X2[~ho2]), y2[~ho2])
p2ho = s2.predict_proba(sc2.transform(X2[ho2]))[:, 1]
S2 = {"n_train_clips_three_tree": int(len(y2)), "n_three": int((y2 == 1).sum()), "n_tree": int((y2 == 0).sum()),
      "holdout_auc": round(float(roc_auc_score(y2[ho2], p2ho)), 4),
      "holdout_acc_at_0.5": round(float(((p2ho >= 0.5).astype(int) == y2[ho2]).mean()), 4)}
np.savez(os.path.join(BASE, "s2_model.npz"), coef=s2.coef_[0], intercept=s2.intercept_,
         scaler_mean=sc2.mean_, scaler_scale=sc2.scale_)
print("S2 train:", S2, flush=True)

def s2_eval(P, y, melarr, tag):
    order = np.argsort(-P, axis=1)
    trig = np.array([set(o[:2].tolist()) == {i3, it} for o in order])
    pred = P.argmax(1).copy(); corr = 0; broke = 0; fired = 0
    idxs = np.where(trig & np.isin(y, [i3, it]))[0]
    for i in idxs:
        mel = melarr[i]
        band_mean = mel.mean(axis=1); band_std = mel.std(axis=1)
        t3 = np.array([mel[:, :21].mean(), mel[:, 21:42].mean(), mel[:, 42:].mean()])
        hi = mel[24:, :].mean(axis=0)
        fr = np.array([hi[:21].mean() - hi[21:].mean(), mel[:, :10].mean(), mel[:, -10:].mean()])
        x = np.concatenate([band_mean, band_std, t3, fr, [P[i, i3], P[i, it], P[i, i3] - P[i, it]]]).astype(np.float32)
        p3 = s2.predict_proba(sc2.transform(x[None, :]))[0, 1]
        new = i3 if p3 >= 0.5 else it
        fired += 1
        if new == y[i] and pred[i] != y[i]: corr += 1
        if new != y[i] and pred[i] == y[i]: broke += 1
        pred[i] = new
    return {"tag": tag, "n_triggered_true_three_tree": int(len(idxs)),
            "corrected": int(corr), "broke": int(broke),
            "v2_argmax_on_triggered": round(float((P.argmax(1)[idxs] == y[idxs]).mean()), 4) if len(idxs) else None,
            "s2_argmax_on_triggered": round(float((pred[idxs] == y[idxs]).mean()), 4) if len(idxs) else None}

S2["child_dev_DEVELOPMENT"] = s2_eval(Pd, yd, meld, "child_dev")
S2["adult_val"] = s2_eval(Pav, yav, melav, "adult_val_both_directions")
print("S2 dev:", S2["child_dev_DEVELOPMENT"], flush=True)
print("S2 adult:", S2["adult_val"], flush=True)
with open(os.path.join(BASE, "metrics_s2.json"), "w") as f:
    json.dump(S2, f, indent=1)
print("S2_DONE", flush=True)
print("SPECIALISTS_DONE", flush=True)

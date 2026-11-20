"""EXPLORATORY assembly diagnostic (NOT a gate, NOT a candidate):
View 3 warp (alpha=0.87) -> frozen v2 int8 -> S1 rescue (F0-gated) -> S2,
scored on JIBO dev only. S1/S2 were trained on v2 probs from unwarped
inputs, so applying them after an inverse-VTLP warp is distribution-
shifted; this measures whether the pieces compose at all, on dev.
Output: metrics_assembly_dev.json.
"""
import os, json, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CE = os.path.join(BASE, "child_eval")
sys.path.insert(0, BASE)
from build_cache_v6 import warp_matrix
import onnxruntime as ort

_m = json.load(open(os.path.join(BASE, "metrics_v2.json")))
CLASSES = _m["classes"]; CIDX = {c: i for i, c in enumerate(CLASSES)}
MU, SD = _m["mel_mean"], _m["mel_std"]
NC = len(CLASSES)
WORD_IDX = np.array([i for i, c in enumerate(CLASSES) if not c.startswith("_")])
TAU = 0.30

def pair_features(mel, P, e):
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

class Logit:
    def __init__(self, z):
        self.c = z["coef"]; self.b = float(z["intercept"][0]) if z["intercept"].ndim else float(z["intercept"])
        self.mu = z["scaler_mean"]; self.sd = z["scaler_scale"]
    def prob(self, X):
        Xs = (X - self.mu) / self.sd
        zz = Xs @ self.c + self.b
        return 1 / (1 + np.exp(-zz))

sess = ort.InferenceSession(os.path.join(BASE, "model_v2_int8.onnx"), providers=["CPUExecutionProvider"])
def probs(M):
    outs = []
    for i in range(0, len(M), 512):
        xb = ((M[i:i + 512] - MU) / SD).astype(np.float32)[:, None, :, :]
        outs.append(sess.run(None, {"mel": xb})[0])
    lg = np.concatenate(outs); lg = lg - lg.max(axis=1, keepdims=True)
    P = np.exp(lg); P /= P.sum(axis=1, keepdims=True)
    return P

def accept_matrix(P):
    top = np.argsort(P, axis=1); t1, t2 = top[:, -1], top[:, -2]
    p1 = P[np.arange(len(P)), t1]; p2 = P[np.arange(len(P)), t2]
    maxother = np.where(np.arange(NC)[None, :] == t1[:, None], p2[:, None], p1[:, None])
    return (P >= TAU) & ((P - maxother) >= 0.0)

dm = np.load(os.path.join(CE, "dev_mel.npz"), allow_pickle=True)
df0 = np.load(os.path.join(CE, "dev_f0.npz"), allow_pickle=True)
X = dm["mel"].astype(np.float32); y = dm["y"]; f0 = df0["F"][:, 0]
Xw = np.einsum("oi,bit->bot", warp_matrix(0.87), X).astype(np.float32)
Pw = probs(Xw)

s1z = np.load(os.path.join(BASE, "s1_model.npz")); S1M = Logit(s1z); thr = float(s1z["threshold"][0])
s2z = np.load(os.path.join(BASE, "s2_model.npz")); S2M = Logit(s2z)

base = accept_matrix(Pw); order = np.argsort(-Pw, axis=1)
extra = np.zeros_like(base); cand = []
for i in range(len(y)):
    for e in order[i, :2]:
        if e in WORD_IDX and not base[i, e] and f0[i] >= 220: cand.append((i, int(e)))
Xc = np.stack([pair_features(Xw[i], Pw[i], e) for i, e in cand])
vp = S1M.prob(Xc)
for k, (i, e) in enumerate(cand):
    if vp[k] >= thr: extra[i, e] = True

acc = base | extra
nt = np.ones_like(acc); nt[np.arange(len(y)), y] = False
wm = np.isin(np.arange(NC), WORD_IDX)[None, :]; mw = np.isin(y, WORD_IDX)
R = {"TA_v2_on_warped": round(float(base[mw, y[mw]].mean()), 4),
     "FA_v2_on_warped": round(float(base[nt & wm].mean()), 4),
     "TA_warped_plus_S1": round(float(acc[mw, y[mw]].mean()), 4),
     "FA_warped_plus_S1": round(float(acc[nt & wm].mean()), 4),
     "argmax_v2_on_warped": round(float((Pw.argmax(1) == y).mean()), 4),
     "corrected_argmax_plus_S1": round(float(((Pw.argmax(1) == y) | extra[np.arange(len(y)), y]).mean()), 4)}

i3, it = CIDX["three"], CIDX["tree"]
pred = Pw.argmax(1).copy(); corr = broke = 0
for i in range(len(y)):
    if set(order[i, :2].tolist()) == {i3, it} and y[i] in (i3, it):
        mel = Xw[i]; bm = mel.mean(1); bs = mel.std(1)
        t3 = np.array([mel[:, :21].mean(), mel[:, 21:42].mean(), mel[:, 42:].mean()])
        hi = mel[24:, :].mean(0)
        fr = np.array([hi[:21].mean() - hi[21:].mean(), mel[:, :10].mean(), mel[:, -10:].mean()])
        x = np.concatenate([bm, bs, t3, fr, [Pw[i, i3], Pw[i, it], Pw[i, i3] - Pw[i, it]]]).astype(np.float32)
        new = i3 if S2M.prob(x[None, :])[0] >= 0.5 else it
        if new == y[i] and pred[i] != y[i]: corr += 1
        if new != y[i] and pred[i] == y[i]: broke += 1
        pred[i] = new
R["s2_corrected"] = int(corr); R["s2_broke"] = int(broke)
R["argmax_warped_plus_S1_S2"] = round(float((pred == y).mean()), 4)
R["reference_unwarped_v2"] = {"argmax": 0.4497, "true_accept": 0.3894, "false_accept": 0.0054}
R["note"] = ("EXPLORATORY assembly on dev: View3 alpha=0.87 warp -> frozen v2 -> S1 (gated) -> S2. "
             "S1/S2 trained on unwarped-input v2 probs: distribution-shifted diagnostic, not a gate, not a candidate.")
print(json.dumps(R, indent=1), flush=True)
json.dump(R, open(os.path.join(BASE, "metrics_assembly_dev.json"), "w"), indent=1)
print("ASSEMBLY_DONE", flush=True)

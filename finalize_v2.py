"""One-shot v2 finalization. Run EXACTLY ONCE for the v2 model version:
1. Load ckpt_v2/best.pt.
2. Tune the verifier decision rule on VALIDATION ONLY:
   accept expected word e iff p(e) >= tau AND p(e) - max_{c != e} p(c) >= margin,
   grid-searched to maximize true-accept subject to validation false-accept <= 0.10
   (false-accept averaged over all (clip, expected != true) pairs).
3. Evaluate the test set ONCE (fp32), plus the int8 ONNX in the same pass.
4. Export model_v2.onnx / model_v2_int8.onnx and write metrics_v2.json.
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn
torch.set_num_threads(4)

BASE = os.path.dirname(os.path.abspath(__file__))
CLASSES = list(np.load(os.path.join(BASE, "classes_v2.npy")))
NC = len(CLASSES)
WORD_IDX = [i for i, c in enumerate(CLASSES) if not c.startswith("_")]

class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
    def forward(self, x):
        x = torch.relu(self.bn1(self.dw(x)))
        return torch.relu(self.bn2(self.pw(x)))

class Net(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.b1 = DSBlock(32, 48); self.b2 = DSBlock(48, 64); self.b3 = DSBlock(64, 96, stride=2)
        self.b4 = DSBlock(96, 96); self.b5 = DSBlock(96, 128, stride=2)
        self.fc = nn.Linear(128, nc)
    def forward(self, x):
        x = self.stem(x)
        for b in [self.b1, self.b2, self.b3, self.b4, self.b5]: x = b(x)
        return self.fc(x.mean(dim=(2, 3)))

ck = torch.load(os.path.join(BASE, "ckpt_v2", "best.pt"))
mu, sd = ck["mu"], ck["sd"]
net = Net(NC); net.load_state_dict(ck["state"]); net.eval()
nparam = sum(p.numel() for p in net.parameters())

def load(split):
    d = np.load(os.path.join(BASE, f"cacheV2_{split}.npz"))
    return d["mel"].astype(np.float32), d["label"], d["source"]

def probs_torch(X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = (torch.from_numpy(X[i:i+512]) - mu) / sd
            lg = net(xb.unsqueeze(1)).numpy()
            lg = lg - lg.max(axis=1, keepdims=True)
            p = np.exp(lg); out.append(p / p.sum(axis=1, keepdims=True))
    return np.concatenate(out)

def rule_stats(P, y, tau, margin):
    """Return (false_accept_rate over e != y pairs, true_accept_rate over y in words with e = y)."""
    top = np.argsort(P, axis=1)
    t1 = top[:, -1]; t2 = top[:, -2]
    p1 = P[np.arange(len(P)), t1]; p2 = P[np.arange(len(P)), t2]
    # max prob over c != e for every e: p1 unless e == t1, then p2
    maxother = np.where(np.arange(NC)[None, :] == t1[:, None], p2[:, None], p1[:, None])
    accept = (P >= tau) & ((P - maxother) >= margin)  # (N, NC) accept for each candidate e
    is_word = np.isin(np.arange(NC), WORD_IDX)
    not_true = np.ones_like(accept); not_true[np.arange(len(y)), y] = False
    fa_mask = not_true & is_word[None, :]
    fa = float(accept[fa_mask].mean())
    m = np.isin(y, WORD_IDX)
    ta = float(accept[m, y[m]].mean())
    return fa, ta

# ---- validation-only rule tuning ----
Xva, yva, _ = load("val")
Pva = probs_torch(Xva)
best_rule = None
for tau in np.arange(0.30, 0.951, 0.05):
    for margin in np.arange(0.0, 0.601, 0.05):
        fa, ta = rule_stats(Pva, yva, float(tau), float(margin))
        if fa <= 0.10:
            key = (ta, float(margin), float(tau))
            if best_rule is None or key > best_rule[0]:
                best_rule = (key, float(tau), float(margin), fa, ta)
if best_rule is None:  # budget unreachable on the grid: take min-FA combo
    cand = []
    for tau in np.arange(0.30, 0.951, 0.05):
        for margin in np.arange(0.0, 0.601, 0.05):
            fa, ta = rule_stats(Pva, yva, float(tau), float(margin))
            cand.append((fa, -ta, float(tau), float(margin), ta))
    cand.sort()
    fa0, _, tau0, m0, ta0 = cand[0]
    best_rule = ((ta0, m0, tau0), tau0, m0, fa0, ta0)
_, TAU, MARGIN, FA_VAL, TA_VAL = best_rule
print(f"rule: tau={TAU:.2f} margin={MARGIN:.2f} val FA={FA_VAL:.4f} val true-accept={TA_VAL:.4f}", flush=True)

# ---- export ONNX fp32 + int8 ----
dummy = torch.randn(1, 1, 40, 63)
onnx_path = os.path.join(BASE, "model_v2.onnx")
torch.onnx.export(net.eval(), dummy, onnx_path, input_names=["mel"], output_names=["logits"],
                  dynamic_axes={"mel": {0: "n"}, "logits": {0: "n"}}, opset_version=13)
from onnxruntime.quantization import quantize_dynamic, QuantType
q_path = os.path.join(BASE, "model_v2_int8.onnx")
quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)

# ---- test set: evaluated once, fp32 + int8 in this single pass ----
Xte, yte, ste = load("test")
Pte = probs_torch(Xte)
pred = Pte.argmax(1)
overall = float((pred == yte).mean())
mask_w = np.isin(yte, WORD_IDX)
word_acc = float((pred[mask_w] == yte[mask_w]).mean())
mask_o = ~mask_w
fa_argmax = float(np.isin(pred[mask_o], WORD_IDX).mean()) if mask_o.sum() else 0.0
per_word = {}
for i, c in enumerate(CLASSES):
    m = yte == i
    if m.sum(): per_word[c] = {"n": int(m.sum()), "acc": round(float((pred[m] == i).mean()), 4)}
per_source = {}
for s in sorted(set(ste.tolist())):
    m = ste == s
    per_source[s] = round(float((pred[m] == yte[m]).mean()), 4)
fa_test, ta_test = rule_stats(Pte, yte, TAU, MARGIN)
# rule FA restricted to _other_/_silence_ clips (v1-metric population)
top = np.argsort(Pte, axis=1); t1 = top[:, -1]; t2 = top[:, -2]
p1 = Pte[np.arange(len(Pte)), t1]; p2 = Pte[np.arange(len(Pte)), t2]
maxother = np.where(np.arange(NC)[None, :] == t1[:, None], p2[:, None], p1[:, None])
accept = (Pte >= TAU) & ((Pte - maxother) >= MARGIN)
fa_os = float(accept[np.ix_(mask_o, WORD_IDX)].mean()) if mask_o.sum() else 0.0

import onnxruntime as ort
sess = ort.InferenceSession(q_path, providers=["CPUExecutionProvider"])
outs = []
for i in range(0, len(Xte), 512):
    xb = ((Xte[i:i+512] - mu) / sd).astype(np.float32)[:, None, :, :]
    outs.append(sess.run(None, {"mel": xb})[0])
lg = np.concatenate(outs); lg = lg - lg.max(axis=1, keepdims=True)
Pq = np.exp(lg); Pq /= Pq.sum(axis=1, keepdims=True)
pred_q = Pq.argmax(1)
int8_acc = float((pred_q == yte).mean())
fa_q, ta_q = rule_stats(Pq, yte, TAU, MARGIN)

metrics = {
    "version": 2,
    "classes": CLASSES,
    "n_vocab_words": len(WORD_IDX),
    "params": nparam,
    "best_epoch": int(ck.get("epoch", -1)),
    "val_best_acc": round(float(ck.get("val_acc", 0.0)), 4),
    "per_source_test_acc": per_source,
    "test_overall_acc": round(overall, 4),
    "test_word_only_acc": round(word_acc, 4),
    "test_false_accept_rate_other_silence_as_word_argmax": round(fa_argmax, 4),
    "decision_rule": {
        "form": "accept expected word e iff p(e) >= tau AND p(e) - max_{c != e} p(c) >= margin",
        "tau": TAU, "margin": MARGIN, "false_accept_budget": 0.10,
        "val_false_accept": round(FA_VAL, 4), "val_true_accept": round(TA_VAL, 4),
        "test_false_accept_all_pairs": round(fa_test, 4),
        "test_false_accept_other_silence": round(fa_os, 4),
        "test_true_accept": round(ta_test, 4),
        "test_false_reject": round(1.0 - ta_test, 4),
    },
    "int8_onnx": {"test_overall_acc": round(int8_acc, 4),
                  "test_false_accept_all_pairs": round(fa_q, 4),
                  "test_true_accept": round(ta_q, 4)},
    "per_word_test": per_word,
    "onnx_fp32_bytes": os.path.getsize(onnx_path),
    "onnx_int8_bytes": os.path.getsize(q_path),
    "mel_mean": float(mu), "mel_std": float(sd),
    "validation_label": "validated on adult/teen voices, not yet on children 5-8",
}
with open(os.path.join(BASE, "metrics_v2.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2), flush=True)
print("FINALIZE_V2_DONE", flush=True)

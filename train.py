"""Train tiny DS-CNN word verifier on cached log-mel. CPU only."""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn
torch.set_num_threads(2)

BASE = os.path.dirname(os.path.abspath(__file__))
EPOCHS = int(os.environ.get("EPOCHS", "14"))
MERGED = os.path.exists(os.path.join(BASE, "cacheM_train.npz"))
if MERGED:
    CLASSES = list(np.load(os.path.join(BASE, "classes_final.npy")))
    PREFIX = "cacheM_"
else:
    CLASSES = list(np.load(os.path.join(BASE, "classes_sc.npy")))
    PREFIX = "cache_sc_"
NC = len(CLASSES)

def load(split):
    d = np.load(os.path.join(BASE, f"{PREFIX}{split}.npz"))
    return d["mel"].astype(np.float32), d["label"]

Xtr, ytr = load("train"); Xva, yva = load("val"); Xte, yte = load("test")
_over = int(os.environ.get("LS_OVER", "0"))
if _over > 1 and os.path.exists(os.path.join(BASE, "cacheM_train.npz")):
    _src = np.load(os.path.join(BASE, "cacheM_train.npz"))["source"]
    _li = np.where(_src == "librispeech")[0]
    _rep = np.tile(_li, _over - 1)
    Xtr = np.concatenate([Xtr, Xtr[_rep]]); ytr = np.concatenate([ytr, ytr[_rep]])
    print(f"oversampled librispeech x{_over}: train -> {len(ytr)}", flush=True)
mu, sd = float(Xtr.mean()), float(Xtr.std())
print(f"train {Xtr.shape} val {Xva.shape} test {Xte.shape} mu {mu:.3f} sd {sd:.3f}", flush=True)

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
        self.b1 = DSBlock(32, 48)
        self.b2 = DSBlock(48, 64)
        self.b3 = DSBlock(64, 96, stride=2)
        self.b4 = DSBlock(96, 96)
        self.b5 = DSBlock(96, 128, stride=2)
        self.fc = nn.Linear(128, nc)
    def forward(self, x):
        x = self.stem(x)
        for b in [self.b1, self.b2, self.b3, self.b4, self.b5]: x = b(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)

net = Net(NC)
nparam = sum(p.numel() for p in net.parameters())
print("params", nparam, flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

def augment(x):
    # Vectorized SpecAugment-lite on batch (B,1,F,T)
    B, _, F, T = x.shape
    dev = x.device
    rows = torch.arange(F, device=dev).view(1, 1, F, 1)
    cols = torch.arange(T, device=dev).view(1, 1, 1, T)
    f0 = torch.randint(0, F-6, (B, 1, 1, 1), device=dev); fw = torch.randint(1, 7, (B, 1, 1, 1), device=dev)
    on = (torch.rand(B, 1, 1, 1, device=dev) < 0.7)
    x = torch.where(on & (rows >= f0) & (rows < f0+fw), torch.full_like(x, mu), x)
    t0 = torch.randint(0, T-8, (B, 1, 1, 1), device=dev); tw = torch.randint(1, 9, (B, 1, 1, 1), device=dev)
    on = (torch.rand(B, 1, 1, 1, device=dev) < 0.7)
    x = torch.where(on & (cols >= t0) & (cols < t0+tw), torch.full_like(x, mu), x)
    if np.random.rand() < 0.5:
        x = torch.roll(x, shifts=int(np.random.randint(-6, 7)), dims=3)
    return x

def run_eval(X, y):
    net.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = (torch.from_numpy(X[i:i+512]) - mu) / sd
            preds.append(net(xb.unsqueeze(1)).argmax(1).numpy())
    return np.concatenate(preds)

best = 0.0; t0 = time.time()
Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
for ep in range(EPOCHS):
    net.train()
    perm = torch.randperm(len(Xtr_t))
    tot = 0.0
    for i in range(0, len(perm), 256):
        idx = perm[i:i+256]
        xb = (Xtr_t[idx] - mu) / sd
        xb = augment(xb.unsqueeze(1).clone())
        loss = lossf(net(xb), ytr_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    sched.step()
    pv = run_eval(Xva, yva); acc = (pv == yva).mean()
    print(f"ep {ep} loss {tot/len(Xtr):.3f} val {acc:.4f} {time.time()-t0:.0f}s", flush=True)
    if acc > best:
        best = acc; torch.save({"state": net.state_dict(), "mu": float(mu), "sd": float(sd)}, os.path.join(BASE, "best.pt"))

ckpt = torch.load(os.path.join(BASE, "best.pt")); net.load_state_dict(ckpt["state"])
pte = run_eval(Xte, yte)
per_word = {}
for i, c in enumerate(CLASSES):
    m = yte == i
    if m.sum(): per_word[c] = {"n": int(m.sum()), "acc": round(float((pte[m] == i).mean()), 4)}
word_idx = [i for i, c in enumerate(CLASSES) if not c.startswith("_")]
mask_w = np.isin(yte, word_idx)
word_acc = float((pte[mask_w] == yte[mask_w]).mean())
mask_o = ~mask_w
false_accept = float(np.isin(pte[mask_o], word_idx).mean()) if mask_o.sum() else 0.0
overall = float((pte == yte).mean())
print("TEST overall", overall, "word-only", word_acc, "false-accept(other/silence->word)", false_accept, flush=True)
print(json.dumps(per_word, indent=1), flush=True)

# export ONNX fp32 + int8 dynamic
dummy = torch.randn(1, 1, 40, Xtr.shape[2])
onnx_path = os.path.join(BASE, "model.onnx")
torch.onnx.export(net.eval(), dummy, onnx_path, input_names=["mel"], output_names=["logits"],
                  dynamic_axes={"mel": {0: "n"}, "logits": {0: "n"}}, opset_version=13)
from onnxruntime.quantization import quantize_dynamic, QuantType
q_path = os.path.join(BASE, "model_int8.onnx")
quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
per_source = {}
if MERGED:
    src = np.load(os.path.join(BASE, "cacheM_test.npz"))["source"]
    for s in sorted(set(src.tolist())):
        m = src == s
        per_source[s] = round(float((pte[m] == yte[m]).mean()), 4)
    print("per-source test acc", per_source, flush=True)

metrics = {
    "classes": CLASSES, "params": nparam, "per_source_test_acc": per_source,
    "val_best": round(float(best), 4), "test_overall_acc": round(overall, 4),
    "test_word_only_acc": round(word_acc, 4),
    "test_false_accept_rate_other_silence_as_word": round(false_accept, 4),
    "per_word_test": per_word,
    "train_seconds": round(time.time()-t0, 1),
    "onnx_fp32_bytes": os.path.getsize(onnx_path), "onnx_int8_bytes": os.path.getsize(q_path),
    "mel_mean": float(mu), "mel_std": float(sd),
}
with open(os.path.join(BASE, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2)
print(json.dumps({k: metrics[k] for k in ["onnx_fp32_bytes", "onnx_int8_bytes"]}), flush=True)
print("DONE", flush=True)

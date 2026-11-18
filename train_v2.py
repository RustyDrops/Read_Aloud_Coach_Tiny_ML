"""Train v2 DS-CNN on cacheV2_*. Saves a checkpoint EVERY epoch to ckpt_v2/
(ep{nn}.pt + last.pt with optimizer state for resume + best.pt by val acc).
Does NOT touch the test set — final evaluation lives in finalize_v2.py.
Env: EPOCHS (default 24), LS_OVER_V2 (default 2), RESUME=1 to continue.
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn
torch.set_num_threads(4)

BASE = os.path.dirname(os.path.abspath(__file__))
EPOCHS = int(os.environ.get("EPOCHS", "16"))
CLASSES = list(np.load(os.path.join(BASE, "classes_v2.npy")))
NC = len(CLASSES)
CKPT_DIR = os.path.join(BASE, "ckpt_v2")
os.makedirs(CKPT_DIR, exist_ok=True)

def load(split):
    d = np.load(os.path.join(BASE, f"cacheV2_{split}.npz"))
    return d["mel"].astype(np.float32), d["label"]

Xtr, ytr = load("train"); Xva, yva = load("val")
_over = int(os.environ.get("LS_OVER_V2", "2"))
if _over > 1:
    _src = np.load(os.path.join(BASE, "cacheV2_train.npz"))["source"]
    _li = np.where(_src == "librispeech")[0]
    _rep = np.tile(_li, _over - 1)
    Xtr = np.concatenate([Xtr, Xtr[_rep]]); ytr = np.concatenate([ytr, ytr[_rep]])
    print(f"oversampled librispeech x{_over}: train -> {len(ytr)}", flush=True)
mu, sd = float(Xtr.mean()), float(Xtr.std())
print(f"train {Xtr.shape} val {Xva.shape} mu {mu:.3f} sd {sd:.3f} classes {NC}", flush=True)

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

start_ep = 0; best = 0.0
last_path = os.path.join(CKPT_DIR, "last.pt")
if os.environ.get("RESUME") == "1" and os.path.exists(last_path):
    ck = torch.load(last_path)
    net.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start_ep = ck["epoch"] + 1; best = ck["best"]
    print(f"resumed at epoch {start_ep} best {best:.4f}", flush=True)

def augment(x):
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

t0 = time.time()
Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
del Xtr
for ep in range(start_ep, EPOCHS):
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
    pv = run_eval(Xva, yva); acc = float((pv == yva).mean())
    print(f"ep {ep} loss {tot/len(Xtr_t):.3f} val {acc:.4f} {time.time()-t0:.0f}s", flush=True)
    if acc > best:
        best = acc
        torch.save({"state": net.state_dict(), "mu": float(mu), "sd": float(sd), "epoch": ep, "val_acc": acc},
                   os.path.join(CKPT_DIR, "best.pt"))
    torch.save({"state": net.state_dict(), "mu": float(mu), "sd": float(sd), "epoch": ep, "val_acc": acc},
               os.path.join(CKPT_DIR, f"ep{ep:02d}.pt"))
    torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": ep, "best": best}, last_path)
print("TRAIN_V2_DONE best val", best, flush=True)

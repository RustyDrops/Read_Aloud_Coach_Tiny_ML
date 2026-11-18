"""Export best.pt -> ONNX fp32 + int8, evaluate per-word test accuracy, write metrics.json.
Works with whichever cache train.py used (cacheM_* preferred, else cache_sc_*)."""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))

merged = os.path.exists(os.path.join(BASE, "cacheM_test.npz"))
if merged:
    dte = np.load(os.path.join(BASE, "cacheM_test.npz")); classes = list(np.load(os.path.join(BASE, "classes_final.npy")))
    Xte, yte = dte["mel"].astype(np.float32), dte["label"]
    src = dte["source"] if "source" in dte else None
else:
    dte = np.load(os.path.join(BASE, "cache_sc_test.npz")); classes = list(np.load(os.path.join(BASE, "classes_sc.npy")))
    Xte, yte = dte["mel"].astype(np.float32), dte["label"]; src = None

import torch.nn as nn
class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout); self.act = nn.ReLU()
    def forward(self, x): return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))
class Net(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.b1 = DSBlock(32, 48); self.b2 = DSBlock(48, 64)
        self.b3 = DSBlock(64, 96, stride=2); self.b4 = DSBlock(96, 96)
        self.b5 = DSBlock(96, 128, stride=2)
        self.fc = nn.Linear(128, ncls)
    def forward(self, x):
        x = self.stem(x); x = self.b2(self.b1(x)); x = self.b4(self.b3(x)); x = self.b5(x)
        return self.fc(x.mean(dim=(2, 3)))

ck = torch.load(os.path.join(BASE, "best.pt"), map_location="cpu", weights_only=False)
mu, sd = ck["mu"], ck["sd"]
model = Net(len(classes)); model.load_state_dict(ck["state"]); model.eval()

def run(X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.from_numpy((X[i:i+512] - mu) / sd).unsqueeze(1)
            out.append(model(xb).argmax(1).numpy())
    return np.concatenate(out)

pred = run(Xte)
metrics = {"classes": classes, "test_n": int(len(yte)), "mu": mu, "sd": sd,
           "note": "validated on adult/teen voices, not yet on children 5-8"}
metrics["overall_acc"] = float((pred == yte).mean())
wi = [i for i, c in enumerate(classes) if not c.startswith("_")]
mask = np.isin(yte, wi)
metrics["word_only_acc"] = float((pred[mask] == yte[mask]).mean())
metrics["per_word_acc"] = {classes[i]: (float((pred[yte == i] == i).mean()) if (yte == i).any() else None) for i in wi}
metrics["per_word_n"] = {classes[i]: int((yte == i).sum()) for i in wi}
oi = [i for i, c in enumerate(classes) if c.startswith("_")]
omask = np.isin(yte, oi)
metrics["false_accept_rate_other_silence"] = float(np.isin(pred[omask], wi).mean()) if omask.any() else None
if src is not None:
    metrics["per_source_acc"] = {s: float((pred[src == s] == yte[src == s]).mean()) for s in sorted(set(src.tolist()))}

dummy = torch.randn(1, 1, 40, 63)
torch.onnx.export(model, dummy, os.path.join(BASE, "model.onnx"), input_names=["mel"], output_names=["logits"],
                  dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}}, opset_version=13)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(os.path.join(BASE, "model.onnx"), os.path.join(BASE, "model_int8.onnx"), weight_type=QuantType.QInt8)
metrics["onnx_fp32_bytes"] = os.path.getsize(os.path.join(BASE, "model.onnx"))
metrics["onnx_int8_bytes"] = os.path.getsize(os.path.join(BASE, "model_int8.onnx"))
with open(os.path.join(BASE, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=1)
print(json.dumps({k: metrics[k] for k in ["overall_acc", "word_only_acc", "false_accept_rate_other_silence", "onnx_fp32_bytes", "onnx_int8_bytes"]}, indent=1))
print("EXPORT DONE")

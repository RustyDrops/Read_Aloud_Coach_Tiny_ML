"""Read-Aloud Coach v2 — verify a spoken clip against an expected page word.

Usage: python predict.py <wav> <expected_word>

Model: model_v2_int8.onnx (42 classes = 40 covered vocabulary words +
_other_ + _silence_). Class list and mel normalisation stats are read from
metrics_v2.json. Both files are looked for next to this script first, then
in the parent directory.

Decision rule (tuned on the validation split only, false-accept budget
<= 0.10; see metrics_v2.json / PROVENANCE.md):
    READ  iff  p(expected) >= TAU  and  p(expected) - max_{c != expected} p(c) >= MARGIN
with TAU = 0.30, MARGIN = 0.0. Otherwise the verdict is NOT THIS WORD when
the model's top prediction is a different covered word, and STUMBLED when
it cannot confirm the word (top is _other_/_silence_, or the expected word
leads but below the confidence threshold).

Validation status: validated on adult/teen voices, not yet on children 5-8.
Treat verdicts as hints for a supervising adult, not grades.
"""
import sys, os, json
import numpy as np
import soundfile as sf
import torch, torchaudio
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
TAU = 0.30
MARGIN = 0.0

def find(name):
    for d in (HERE, os.path.dirname(HERE)):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise SystemExit(f"cannot find {name} next to predict.py or in its parent directory")

metrics = json.load(open(find("metrics_v2.json")))
CLASSES = metrics["classes"]
mu, sd = metrics["mel_mean"], metrics["mel_std"]
mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=512, hop_length=256, n_mels=40)

def main(wav_path, expected):
    x, sr = sf.read(wav_path, dtype="float32")
    if sr != SR:
        raise SystemExit(f"need 16kHz wav, got {sr}")
    if x.ndim > 1:
        x = x[:, 0]
    if len(x) > SR:
        x = x[:SR]
    elif len(x) < SR:
        x = np.pad(x, (0, SR - len(x)))
    m = torch.log(mel_tf(torch.from_numpy(x).unsqueeze(0)) + 1e-6).numpy()
    m = ((m - mu) / sd).astype(np.float32)[None]
    sess = ort.InferenceSession(find("model_v2_int8.onnx"), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"mel": m})[0][0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    top = int(np.argmax(p))
    print(f"predicted: {CLASSES[top]} ({p[top]:.2f})")
    if expected not in CLASSES:
        covered = [c for c in CLASSES if not c.startswith("_")]
        print(f"'{expected}' is not in the covered vocabulary ({len(covered)} words): {covered}")
        return
    exp_i = CLASSES.index(expected)
    p_exp = float(p[exp_i])
    max_other = float(np.max(np.delete(p, exp_i)))
    print(f"expected '{expected}' probability: {p_exp:.2f}")
    if p_exp >= TAU and (p_exp - max_other) >= MARGIN:
        print("verdict: READ")
    elif top != exp_i and not CLASSES[top].startswith("_"):
        print(f"verdict: NOT THIS WORD (heard '{CLASSES[top]}')")
    else:
        print("verdict: STUMBLED / NOT THIS WORD")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])

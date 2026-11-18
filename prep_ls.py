"""Extract isolated vocab-word clips from LibriSpeech dev-clean via CTC forced alignment
(labels come from the human transcripts; alignment only locates them).
Merges with the Speech Commands cache into cacheM_{split}.npz with unified classes.
Speaker-disjoint split by LibriSpeech speaker id (hash % 10: 0-7 train, 8 val, 9 test).
"""
import os, sys, glob, tarfile, hashlib
import numpy as np
import soundfile as sf
import torch, torchaudio
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab import VOCAB_50, SC_COVERED

BASE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
LS_DIR = os.path.join(BASE, "data", "LibriSpeech", "dev-clean")
if not os.path.isdir(LS_DIR):
    with tarfile.open(os.path.join(BASE, "data", "dev-clean.tar.gz")) as t:
        t.extractall(os.path.join(BASE, "data"))
    print("extracted LibriSpeech", flush=True)

TARGETS = [w for w in VOCAB_50 if w not in SC_COVERED]
print("LS target words:", TARGETS, flush=True)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
labels = bundle.get_labels()
lab2idx = {c: i for i, c in enumerate(labels)}
model = bundle.get_model(); model.eval()

mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=512, hop_length=256, n_mels=40)
def to_mel(waves):
    out = []
    for i in range(0, len(waves), 512):
        m = mel_tf(torch.from_numpy(np.stack(waves[i:i+512])))
        out.append(torch.log(m + 1e-6).half().numpy())
    return np.concatenate(out)

# gather short utterances
utts = []  # (flac_path, transcript, speaker)
for tf in glob.glob(os.path.join(LS_DIR, "*", "*", "*.trans.txt")):
    spk = tf.split(os.sep)[-3]
    for line in open(tf):
        uid, txt = line.strip().split(" ", 1)
        p = os.path.join(os.path.dirname(tf), uid + ".flac")
        info = sf.info(p)
        if info.frames / info.samplerate <= 12.0:
            utts.append((p, txt, spk))
utts.sort(key=lambda x: hashlib.md5(x[0].encode()).hexdigest())
_skip = int(os.environ.get("LS_SKIP", "0"))
utts = utts[_skip:_skip + int(os.environ.get("LS_UTTS", "600"))]
print("utterances:", len(utts), flush=True)

def align_words(wave, transcript):
    """Return list of (word, start_s, end_s) via forced alignment."""
    words = transcript.lower().split()
    chars = []
    word_tok_spans = []  # (word, first_tok_idx, last_tok_idx) in token seq
    for w in words:
        clean = "".join(c for c in w if c.isalpha() or c == "'")
        if not clean: continue
        start = len(chars)
        for c in clean.upper():
            if c not in lab2idx: return []
            chars.append(lab2idx[c])
        word_tok_spans.append((clean, start, len(chars)-1))
        chars.append(lab2idx["|"])
    if chars and chars[-1] == lab2idx["|"]: chars.pop()
    with torch.no_grad():
        em, _ = model(wave.unsqueeze(0))
    logp = torch.log_softmax(em, dim=-1)
    tgt = torch.tensor([chars], dtype=torch.int32)
    try:
        aligned, _ = torchaudio.functional.forced_align(logp.int() if False else logp, tgt, blank=0)
    except Exception as e:
        print("align fail:", e, flush=True); return []
    al = aligned[0].numpy()
    # scan frames assigning to target positions monotonically
    pos_start = {}; pos_end = {}
    j = 0
    for t, a in enumerate(al):
        if a == 0: continue
        while j < len(chars) and a != chars[j]:
            j += 1
        if j >= len(chars): break
        if j not in pos_start: pos_start[j] = t
        pos_end[j] = t
    ratio = wave.shape[0] / len(al)  # samples per emission frame
    out = []
    for w, s, e in word_tok_spans:
        if s in pos_start and e in pos_end:
            out.append((w, pos_start[s]*ratio/SR, (pos_end[e]+1)*ratio/SR))
    return out

CAP = 260; MINN = 40
cache_clips = os.path.join(BASE, "ls_clips.npz")
if os.path.exists(cache_clips):
    d = np.load(cache_clips, allow_pickle=True)
    collected = {w: list(zip(d["waves"][d["words"] == w], d["spks"][d["words"] == w])) for w in set(d["words"].tolist())}
    print("loaded cached LS clips", {w: len(v) for w, v in collected.items()}, flush=True)
else:
  collected = {}  # word -> list of (wave1s, speaker)
if os.path.exists(cache_clips) and int(os.environ.get("LS_SKIP", "0")) == 0: utts = []
DONE_F = os.path.join(BASE, "ls_done.txt")
done_paths = set()
if os.path.exists(DONE_F):
    done_paths = set(x.strip() for x in open(DONE_F) if x.strip())
def save_cache():
    if not collected: return
    _w = np.concatenate([np.stack([c for c, s in v]) for v in collected.values()])
    _lab = np.concatenate([np.array([w]*len(v)) for w, v in collected.items()])
    _spk = np.concatenate([np.array([s for c, s in v]) for v in collected.values()])
    np.savez_compressed(cache_clips, waves=_w, words=_lab, spks=_spk)
    print("saved ls_clips.npz", _w.shape, flush=True)
for _ui, (p, txt, spk) in enumerate(utts):
    if p in done_paths: continue
    if _ui % 10 == 0: print("utt", _ui, "clips", sum(len(v) for v in collected.values()), flush=True)
    x, sr = sf.read(p, dtype="float32")
    spans = align_words(torch.from_numpy(x), txt)
    for w, s, e in spans:
        if w not in TARGETS: continue
        if not (0.04 <= e-s <= 1.15): continue
        seg = x[int(max(0, s-0.04)*SR): int((e+0.04)*SR)]
        if len(seg) >= SR: seg = seg[:SR]
        clip = np.zeros(SR, dtype=np.float32)
        off = (SR - len(seg)) // 2
        clip[off:off+len(seg)] = seg
        collected.setdefault(w, []).append((clip, spk))
    done_words = [w for w, v in collected.items() if len(v) >= CAP]
    if len(done_words) >= len(TARGETS): break
    if _ui % 50 == 49:
        with open(DONE_F, "a") as f: f.write("\n".join(u[0] for u in utts[max(0,_ui-49):_ui+1]) + "\n")
        save_cache()
print({w: len(v) for w, v in sorted(collected.items())}, flush=True)
if collected:
    save_cache()

new_words = sorted([w for w, v in collected.items() if len(v) >= MINN])
print("new words kept (>=40 clips):", new_words, flush=True)
if not new_words:
    print("NO_LS_WORDS", flush=True); sys.exit(0)

CLASSES = SC_COVERED + new_words + ["_other_", "_silence_"]
cidx = {c: i for i, c in enumerate(CLASSES)}
np.save(os.path.join(BASE, "classes_final.npy"), np.array(CLASSES))

for split in ["train", "val", "test"]:
    d = np.load(os.path.join(BASE, f"cache_sc_{split}.npz"))
    sc_mel, sc_lab = d["mel"], d["label"]
    sc_classes = list(np.load(os.path.join(BASE, "classes_sc.npy")))
    remap = np.array([cidx[c] for c in sc_classes])
    mels = [sc_mel]; labs = [remap[sc_lab]]; srcs = [np.array(["speech_commands"]*len(sc_lab))]
    waves, labs_ls = [], []
    for w in new_words:
        items = collected[w][:CAP]
        for clip, spk in items:
            h = int(hashlib.md5(spk.encode()).hexdigest(), 16) % 10
            s = "train" if h <= 7 else ("val" if h == 8 else "test")
            if s == split:
                waves.append(clip); labs_ls.append(cidx[w])
    if waves:
        mels.append(to_mel(waves)); labs.append(np.array(labs_ls))
        srcs.append(np.array(["librispeech"]*len(labs_ls)))
    np.savez_compressed(os.path.join(BASE, f"cacheM_{split}.npz"),
                        mel=np.concatenate(mels), label=np.concatenate(labs), source=np.concatenate(srcs))
    print("merged", split, np.concatenate(mels).shape, flush=True)
print("DONE", flush=True)

"""v2 (optimized): align the REST of LibriSpeech dev-clean.
Same labels discipline as v1: words located by CTC forced alignment against
the corpus's own human transcripts; no model output is ever a label.
Speed vs v1: (a) wav2vec2 forward passes are batched (utterances grouped by
duration, padded, lengths passed so padded frames are masked), (b) utterances
whose transcripts contain no not-yet-saturated target word are skipped
(transcript-based selection only). The first 3 processed utterances are also
run through the v1 single-utterance path and the spans compared (sanity).
Checkpointing: shards ls_shard_v2_{k}.npz + ls_done_v2.txt every flush.
"""
import os, sys, glob, hashlib, time
import numpy as np
import soundfile as sf
import torch, torchaudio
torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab import VOCAB_50, SC_COVERED

BASE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
LS_DIR = os.path.join(BASE, "data", "LibriSpeech", "dev-clean")
TARGETS = set(w for w in VOCAB_50 if w not in SC_COVERED)
SATURATE = 700
BATCH = 8
FLUSH_EVERY = 25  # batches

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
labels = bundle.get_labels()
lab2idx = {c: i for i, c in enumerate(labels)}
model = bundle.get_model(); model.eval()

# pool counts (v1 + any v2 shards already on disk)
pool_counts = {}
d = np.load(os.path.join(BASE, "ls_clips.npz"), allow_pickle=True)
for w in d["words"].tolist(): pool_counts[w] = pool_counts.get(w, 0) + 1
for f in glob.glob(os.path.join(BASE, "ls_shard_v2_*.npz")):
    dd = np.load(f, allow_pickle=True)
    for w in dd["words"].tolist(): pool_counts[w] = pool_counts.get(w, 0) + 1
print("initial pool counts:", dict(sorted(pool_counts.items())), flush=True)

utts = []
for tf in glob.glob(os.path.join(LS_DIR, "*", "*", "*.trans.txt")):
    spk = tf.split(os.sep)[-3]
    for line in open(tf):
        uid, txt = line.strip().split(" ", 1)
        p = os.path.join(os.path.dirname(tf), uid + ".flac")
        info = sf.info(p)
        if info.frames / info.samplerate <= 12.0:
            utts.append((p, txt, spk, info.frames))
utts.sort(key=lambda x: hashlib.md5(x[0].encode()).hexdigest())
done = set()
if os.path.exists(os.path.join(BASE, "ls_done.txt")):
    done |= set(x.strip() for x in open(os.path.join(BASE, "ls_done.txt")) if x.strip())
done |= set(u[0] for u in utts[:500])
DONE_F = os.path.join(BASE, "ls_done_v2.txt")
if os.path.exists(DONE_F):
    done |= set(x.strip() for x in open(DONE_F) if x.strip())
todo = [u for u in utts if u[0] not in done]
todo.sort(key=lambda x: x[3])  # by duration for efficient batching
print(f"todo: {len(todo)} utterances", flush=True)

def transcript_targets(txt):
    out = set()
    for w in txt.lower().split():
        clean = "".join(c for c in w if c.isalpha() or c == "'")
        if clean in TARGETS: out.add(clean)
    return out

def build_targets(transcript):
    words = transcript.lower().split()
    chars = []; spans = []
    for w in words:
        clean = "".join(c for c in w if c.isalpha() or c == "'")
        if not clean: continue
        start = len(chars)
        for c in clean.upper():
            if c not in lab2idx: return None, None
            chars.append(lab2idx[c])
        spans.append((clean, start, len(chars)-1))
        chars.append(lab2idx["|"])
    if chars and chars[-1] == lab2idx["|"]: chars.pop()
    return chars, spans

def spans_from_logp(logp_2d, chars, spans, n_samples):
    tgt = torch.tensor([chars], dtype=torch.int32)
    try:
        aligned, _ = torchaudio.functional.forced_align(logp_2d.unsqueeze(0), tgt, blank=0)
    except Exception as e:
        print("align fail:", e, flush=True); return []
    al = aligned[0].numpy()
    pos_start = {}; pos_end = {}
    j = 0
    for t, a in enumerate(al):
        if a == 0: continue
        while j < len(chars) and a != chars[j]:
            j += 1
        if j >= len(chars): break
        if j not in pos_start: pos_start[j] = t
        pos_end[j] = t
    # Repair zero-frame tokens (see prep_tc100_v2.py for rationale: doubled
    # letters like the EE in "SEE" collapse to one emission run and the
    # strict rule silently dropped such words).
    L = len(chars)
    est = []
    for i in range(L):
        if i in pos_start:
            est.append((pos_start[i], pos_end[i])); continue
        p = next((k for k in range(i-1, -1, -1) if k in pos_end), None)
        n = next((k for k in range(i+1, L) if k in pos_start), None)
        if p is not None and n is not None: v = (pos_end[p] + pos_start[n]) // 2
        elif p is not None: v = pos_end[p] + 1
        elif n is not None: v = max(0, pos_start[n] - 1)
        else: v = 0
        est.append((v, v))
    ratio = n_samples / len(al)
    out = []
    for w, s, e in spans:
        if s not in pos_start and e not in pos_end and (e - s + 1) > 3: continue
        out.append((w, est[s][0]*ratio/SR, (est[e][1]+1)*ratio/SR))
    return out

def extract(x, spans, spk, sink):
    for w, s, e in spans:
        if w not in TARGETS: continue
        if pool_counts.get(w, 0) >= SATURATE: continue
        if not (0.04 <= e-s <= 1.15): continue
        seg = x[int(max(0, s-0.04)*SR): int((e+0.04)*SR)]
        if len(seg) >= SR: seg = seg[:SR]
        clip = np.zeros(SR, dtype=np.float32)
        off = (SR - len(seg)) // 2
        clip[off:off+len(seg)] = seg
        sink.append((clip, w, spk))
        pool_counts[w] = pool_counts.get(w, 0) + 1

existing = glob.glob(os.path.join(BASE, "ls_shard_v2_*.npz"))
shard_k = max([int(os.path.basename(f).split("_")[3].split(".")[0]) for f in existing], default=-1) + 1
new_clips = []; done_batch = []; total_new = 0; skipped = 0
t_start = time.time(); checked = 0
for bi in range(0, len(todo), BATCH):
    batch = todo[bi:bi+BATCH]
    waves = []; metas = []
    for p, txt, spk, nfr in batch:
        done_batch.append(p)
        tg = transcript_targets(txt)
        if not tg or all(pool_counts.get(w, 0) >= SATURATE for w in tg):
            skipped += 1; continue
        chars, spans = build_targets(txt)
        if chars is None: continue
        try:
            x, sr = sf.read(p, dtype="float32")
        except Exception:
            continue
        waves.append(x); metas.append((x, chars, spans, spk))
    if waves:
        L = max(len(w) for w in waves)
        arr = np.zeros((len(waves), L), dtype=np.float32)
        for i, w in enumerate(waves): arr[i, :len(w)] = w
        lens = torch.tensor([len(w) for w in waves], dtype=torch.int64)
        with torch.no_grad():
            em, out_lens = model(torch.from_numpy(arr), lens)
        logp_all = torch.log_softmax(em, dim=-1)
        for i, (x, chars, spans, spk) in enumerate(metas):
            lp = logp_all[i, :int(out_lens[i])]
            sp = spans_from_logp(lp, chars, spans, len(x))
            if checked < 3:
                # sanity: v1 single-utterance path on the same audio
                with torch.no_grad():
                    em1, ol1 = model(torch.from_numpy(x).unsqueeze(0))
                n1 = int(ol1[0]) if ol1 is not None else em1.shape[1]
                lp1 = torch.log_softmax(em1, dim=-1)[0, :n1]
                sp1 = spans_from_logp(lp1, chars, spans, len(x))
                same = len(sp) == len(sp1) and all(a[0] == b[0] and abs(a[1]-b[1]) < 0.03 and abs(a[2]-b[2]) < 0.03 for a, b in zip(sp, sp1))
                print(f"sanity check {checked}: batched {len(sp)} spans vs single {len(sp1)} spans, match={same}", flush=True)
                checked += 1
            extract(x, sp, spk, new_clips)
    if (bi // BATCH + 1) % FLUSH_EVERY == 0 or bi + BATCH >= len(todo):
        if new_clips:
            np.savez_compressed(os.path.join(BASE, f"ls_shard_v2_{shard_k}.npz"),
                                waves=np.stack([c for c, w, s in new_clips]),
                                words=np.array([w for c, w, s in new_clips]),
                                spks=np.array([s for c, w, s in new_clips]))
            total_new += len(new_clips); shard_k += 1; new_clips = []
        with open(DONE_F, "a") as f:
            f.write("\n".join(done_batch) + "\n")
        done_batch = []
        el = time.time() - t_start
        print(f"progress {min(bi+BATCH, len(todo))}/{len(todo)} utts, skipped {skipped}, "
              f"new clips {total_new}, {el/60:.1f} min elapsed, "
              f"pool: {dict(sorted(pool_counts.items()))}", flush=True)
print("ALIGN_V2_DONE total_new_clips", total_new, "skipped", skipped, flush=True)

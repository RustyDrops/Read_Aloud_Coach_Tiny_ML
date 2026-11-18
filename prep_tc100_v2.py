"""v2 top-up: align a bounded slice of LibriSpeech train-clean-100, extracting
ONLY target words still below the coverage bar after the dev-clean pool
(labels from human transcripts via forced alignment, as before).
Skips (without inference) any utterance whose transcript has no wanted word.
Stops when every wanted word reaches GOAL clips or MAX_UTTS is reached.
Shards: ls_shard_tc_{k}.npz; done: ls_done_tc.txt. Speaker split at merge
time uses the same md5%10 rule; train-clean-100 speakers are disjoint from
dev-clean speakers by LibriSpeech design.
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
LS_DIR = os.path.join(BASE, "data", "LibriSpeech", "train-clean-100")
ALL_TARGETS = set(w for w in VOCAB_50 if w not in SC_COVERED)
GOAL = 80
MAX_UTTS = 8000
BATCH = 8
FLUSH_EVERY = 25

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
labels = bundle.get_labels()
lab2idx = {c: i for i, c in enumerate(labels)}
model = bundle.get_model(); model.eval()

pool_counts = {}
d = np.load(os.path.join(BASE, "ls_clips.npz"), allow_pickle=True)
for w in d["words"].tolist(): pool_counts[w] = pool_counts.get(w, 0) + 1
for pat in ["ls_shard_v2_*.npz", "ls_shard_tc_*.npz"]:
    for f in glob.glob(os.path.join(BASE, pat)):
        dd = np.load(f, allow_pickle=True)
        for w in dd["words"].tolist(): pool_counts[w] = pool_counts.get(w, 0) + 1
WANTED = set(w for w in ALL_TARGETS if pool_counts.get(w, 0) < GOAL)
print("wanted words (<80):", sorted(WANTED), flush=True)
print("pool:", dict(sorted(pool_counts.items())), flush=True)

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
DONE_F = os.path.join(BASE, "ls_done_tc.txt")
if os.path.exists(DONE_F):
    done |= set(x.strip() for x in open(DONE_F) if x.strip())
todo = [u for u in utts if u[0] not in done]
todo.sort(key=lambda x: x[3])
print(f"train-clean-100 utterances (<=12s): {len(utts)}, todo {len(todo)}", flush=True)

def transcript_wanted(txt):
    out = set()
    for w in txt.lower().split():
        clean = "".join(c for c in w if c.isalpha() or c == "'")
        if clean in WANTED: out.add(clean)
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
    # Repair zero-frame tokens (e.g. the second E of "SEE" collapses into one
    # emission run, so the alignment path gives it no frames and the strict
    # rule silently drops the word). A missing token's position is estimated
    # as the midpoint between its neighbours' frames; labels still come only
    # from the transcript. Words with NO located token and >3 tokens are
    # still dropped (alignment failure, not a collapse).
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

existing = glob.glob(os.path.join(BASE, "ls_shard_tc_*.npz"))
shard_k = max([int(os.path.basename(f).split("_")[3].split(".")[0]) for f in existing], default=-1) + 1
new_clips = []; done_batch = []; total_new = 0; skipped = 0; processed = 0
t_start = time.time()
stop = False
for bi in range(0, len(todo), BATCH):
    if stop or processed >= MAX_UTTS: break
    batch = todo[bi:bi+BATCH]
    waves = []; metas = []
    for p, txt, spk, nfr in batch:
        done_batch.append(p); processed += 1
        if not (transcript_wanted(txt) - set(w for w in WANTED if pool_counts.get(w, 0) >= GOAL)):
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
            sp = spans_from_logp(logp_all[i, :int(out_lens[i])], chars, spans, len(x))
            for w, s, e in sp:
                if w not in WANTED or pool_counts.get(w, 0) >= GOAL: continue
                if not (0.04 <= e-s <= 1.15): continue
                seg = x[int(max(0, s-0.04)*SR): int((e+0.04)*SR)]
                if len(seg) >= SR: seg = seg[:SR]
                clip = np.zeros(SR, dtype=np.float32)
                off = (SR - len(seg)) // 2
                clip[off:off+len(seg)] = seg
                new_clips.append((clip, w, spk))
                pool_counts[w] = pool_counts.get(w, 0) + 1
    if all(pool_counts.get(w, 0) >= GOAL for w in WANTED):
        stop = True
    if (bi // BATCH + 1) % FLUSH_EVERY == 0 or stop or processed >= MAX_UTTS or bi + BATCH >= len(todo):
        if new_clips:
            np.savez_compressed(os.path.join(BASE, f"ls_shard_tc_{shard_k}.npz"),
                                waves=np.stack([c for c, w, s in new_clips]),
                                words=np.array([w for c, w, s in new_clips]),
                                spks=np.array([s for c, w, s in new_clips]))
            total_new += len(new_clips); shard_k += 1; new_clips = []
        with open(DONE_F, "a") as f:
            f.write("\n".join(done_batch) + "\n")
        done_batch = []
        print(f"progress {processed} utts, skipped {skipped}, new clips {total_new}, "
              f"{(time.time()-t_start)/60:.1f} min, wanted now: "
              f"{ {w: pool_counts.get(w, 0) for w in sorted(WANTED)} }", flush=True)
print("TC100_DONE total_new_clips", total_new, flush=True)

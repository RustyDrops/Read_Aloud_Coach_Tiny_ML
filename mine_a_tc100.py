"""Step 2 data lever (PLAN-VIEWS-SPECIALISTS): mine 'a' at scale from
LibriSpeech train-clean-100 with the project's existing forced-alignment
pipeline (prep_ls_v2.py lineage). Labels come ONLY from the corpus's own
human transcripts + CTC forced alignment (wav2vec2 ASR base 960h) — no
model output is ever a label, same discipline as v1/v2 mining.

Why 'a' was scarce (measured): the pool holds 80 'a' clips vs 700 (cap)
for 'and'/'the'/'to'. The v1/v2 extractor required span duration
>= 0.04 s; a schwa 'a' typically aligns to 1-2 CTC frames (0.02-0.04 s),
so >99% of instances were silently dropped. Documented changes here,
extraction only:
  - min span duration 0.01 s (was 0.04)
  - context padding +-0.08 s (was +-0.04) so the 1 s clip keeps the
    schwa plus a little coarticulation context, centered as before
Everything else (alignment, span repair, 1 s centered clip) is the
project pipeline unchanged.

Scope: train-clean-100 utterances NOT already processed by prep_tc100
(ls_done_tc.txt), transcript contains the token 'a', duration <= 10 s,
sorted shortest-first. Stop at TARGET clips or TIME_BUDGET_S.
Shards: mine_a_shard_{k}.npz (waves, words, spks) + mine_a_done.txt.
"""
import os, sys, glob, hashlib, time
import numpy as np
import soundfile as sf
import torch, torchaudio
torch.set_num_threads(4)

BASE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
LS_DIR = os.path.join(BASE, "data", "LibriSpeech", "train-clean-100")
TARGET = "a"
TARGET_CLIPS = int(os.environ.get("MINE_A_TARGET", "2500"))
TIME_BUDGET_S = float(os.environ.get("MINE_A_TIME", "2700"))
MAX_PER_UTT = 6
BATCH = 8
FLUSH_EVERY = 25

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
labels = bundle.get_labels()
lab2idx = {c: i for i, c in enumerate(labels)}
model = bundle.get_model(); model.eval()

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
        spans.append((clean, start, len(chars) - 1))
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
    L = len(chars)
    est = []
    for i in range(L):
        if i in pos_start:
            est.append((pos_start[i], pos_end[i])); continue
        p = next((k for k in range(i - 1, -1, -1) if k in pos_end), None)
        n = next((k for k in range(i + 1, L) if k in pos_start), None)
        if p is not None and n is not None: v = (pos_end[p] + pos_start[n]) // 2
        elif p is not None: v = pos_end[p] + 1
        elif n is not None: v = max(0, pos_start[n] - 1)
        else: v = 0
        est.append((v, v))
    ratio = n_samples / len(al)
    out = []
    for w, s, e in spans:
        if s not in pos_start and e not in pos_end and (e - s + 1) > 3: continue
        out.append((w, est[s][0] * ratio / SR, (est[e][1] + 1) * ratio / SR))
    return out

# ---- utterance list ----
done = set()
for fn in ["ls_done_tc.txt"]:
    p = os.path.join(BASE, fn)
    if os.path.exists(p):
        done |= set(x.strip() for x in open(p) if x.strip())
DONE_F = os.path.join(BASE, "mine_a_done.txt")
if os.path.exists(DONE_F):
    done |= set(x.strip() for x in open(DONE_F) if x.strip())
utts = []
for tf in glob.glob(os.path.join(LS_DIR, "*", "*", "*.trans.txt")):
    spk = tf.split(os.sep)[-3]
    for line in open(tf):
        uid, txt = line.strip().split(" ", 1)
        toks = ["".join(c for c in w if c.isalpha() or c == "'") for w in txt.lower().split()]
        if TARGET not in toks: continue
        p = os.path.join(os.path.dirname(tf), uid + ".flac")
        if p in done: continue
        info = sf.info(p)
        dur = info.frames / info.samplerate
        if dur <= 10.0:
            utts.append((p, txt, spk, info.frames))
utts.sort(key=lambda x: x[3])  # shortest first
print(f"candidate utterances with 'a': {len(utts)}", flush=True)

existing = glob.glob(os.path.join(BASE, "mine_a_shard_*.npz"))
shard_k = max([int(os.path.basename(f).split("_")[3].split(".")[0]) for f in existing], default=-1) + 1
have = 0
for f in existing:
    have += len(np.load(f, allow_pickle=True)["words"])
print(f"already mined: {have}, next shard {shard_k}", flush=True)

new_clips = []; done_batch = []; total_new = have
t_start = time.time()
stop = False
for bi in range(0, len(utts), BATCH):
    if stop: break
    batch = utts[bi:bi + BATCH]
    waves = []; metas = []
    for p, txt, spk, nfr in batch:
        done_batch.append(p)
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
            got = 0
            for w, s, e in sp:
                if w != TARGET or got >= MAX_PER_UTT: continue
                if not (0.01 <= e - s <= 1.15): continue
                seg = x[int(max(0, s - 0.08) * SR): int((e + 0.08) * SR)]
                if len(seg) >= SR: seg = seg[:SR]
                clip = np.zeros(SR, dtype=np.float32)
                off = (SR - len(seg)) // 2
                clip[off:off + len(seg)] = seg
                new_clips.append((clip, w, spk)); got += 1; total_new += 1
    if (bi // BATCH + 1) % FLUSH_EVERY == 0 or bi + BATCH >= len(utts):
        if new_clips:
            np.savez_compressed(os.path.join(BASE, f"mine_a_shard_{shard_k}.npz"),
                                waves=np.stack([c for c, w, s in new_clips]),
                                words=np.array([w for c, w, s in new_clips]),
                                spks=np.array([s for c, w, s in new_clips]))
            shard_k += 1; new_clips = []
        with open(DONE_F, "a") as f:
            f.write("\n".join(done_batch) + "\n")
        done_batch = []
        print(f"progress {min(bi + BATCH, len(utts))}/{len(utts)} utts, 'a' clips {total_new}, "
              f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    if total_new >= TARGET_CLIPS or time.time() - t_start > TIME_BUDGET_S:
        stop = True
# final flush
if new_clips:
    np.savez_compressed(os.path.join(BASE, f"mine_a_shard_{shard_k}.npz"),
                        waves=np.stack([c for c, w, s in new_clips]),
                        words=np.array([w for c, w, s in new_clips]),
                        spks=np.array([s for c, w, s in new_clips]))
if done_batch:
    with open(DONE_F, "a") as f:
        f.write("\n".join(done_batch) + "\n")
print("MINE_A_DONE total_a_clips", total_new, flush=True)

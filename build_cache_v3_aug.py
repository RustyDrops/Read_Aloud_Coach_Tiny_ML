"""RA-PT1 cache build: originals (mel verified vs cacheV7_train) + one
register-shifted copy per clip (resample x1.26 pitch/tempo, random-start
pad, seed 7). Output: cachePT1_train.npz in cacheV7 format."""
import os, random
import numpy as np
import torch, torchaudio
from scipy.signal import resample_poly

RA = os.path.expanduser('~/workspace/micro-edge-pilot/read-aloud')
D1 = os.path.join(RA, 'autonomous/ra-d1')
HERE = os.path.join(RA, 'autonomous/ra-pt1')
SR = 16000
mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=512, hop_length=256, n_mels=40)

def to_mel(waves):
    out = []
    for i in range(0, len(waves), 512):
        m = mel_tf(torch.from_numpy(np.ascontiguousarray(waves[i:i+512])))
        out.append(torch.log(m + 1e-6).numpy())
    return np.concatenate(out)

sc = np.load(os.path.join(D1, 'sc_train_waves.npy'), mmap_mode='r')
ls = np.load(os.path.join(D1, 'ls_train_waves.npy'), mmap_mode='r')
labels = np.concatenate([np.load(os.path.join(D1, 'sc_train_labels.npy')),
                         np.load(os.path.join(D1, 'ls_train_labels.npy'))])
srcs = np.array(['speech_commands'] * len(sc) + ['librispeech'] * len(ls))
cache = np.load(os.path.join(RA, 'cacheV7_train.npz'))
assert (cache['label'] == labels).all() and (cache['source'] == srcs).all()
N = len(labels)
print('n', N, flush=True)

rng = random.Random(7)
new_len = int(SR / 1.26)
mel_all = np.zeros((2 * N, 40, 63), dtype=np.float16)
ok = 0
for s0 in range(0, N, 2000):
    s1 = min(N, s0 + 2000)
    idx = np.arange(s0, s1)
    w = np.empty((len(idx), SR), dtype=np.float32)
    m_sc = idx < len(sc)
    if m_sc.any(): w[m_sc] = np.asarray(sc[idx[m_sc]])
    if (~m_sc).any(): w[~m_sc] = np.asarray(ls[idx[~m_sc] - len(sc)])
    m0 = to_mel(w)
    diff = np.abs(m0 - cache['mel'][s0:s1].astype(np.float32))
    ok += int((diff.max(axis=(1, 2)) < 0.6).sum())
    mel_all[s0:s1] = m0.astype(np.float16)
    pitched = np.zeros_like(w)
    for i in range(len(w)):
        y = resample_poly(w[i], SR, new_len).astype(np.float32)
        if len(y) < SR:
            st = rng.randrange(0, SR - len(y) + 1)
            pitched[i, st:st+len(y)] = y
        else:
            pitched[i] = y[:SR]
    mel_all[N+s0:N+s1] = to_mel(pitched).astype(np.float16)
    print(s1, '/', N, 'orig frac ok', ok / s1, flush=True)
assert ok / N >= 0.99, 'originals verification failed'
np.savez_compressed(os.path.join(HERE, 'cachePT1_train.npz'),
                    mel=mel_all, label=np.concatenate([labels, labels]),
                    source=np.concatenate([srcs, srcs]))
print('saved cachePT1_train', mel_all.shape, flush=True)
print('BUILD_PT1_DONE', flush=True)

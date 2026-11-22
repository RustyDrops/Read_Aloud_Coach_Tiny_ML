"""RA-C1 cache: PT1 cache + VTLP copies of v7 originals + PH2 phrase copies."""
import os, sys
import numpy as np
RA = os.path.expanduser('~/workspace/micro-edge-pilot/read-aloud')
sys.path.insert(0, RA)
from build_cache_v6 import warp_matrix
HERE = os.path.join(RA, 'autonomous/ra-c1')
pt1 = np.load(os.path.join(RA, 'autonomous/ra-pt1/cachePT1_train.npz'))
v7 = np.load(os.path.join(RA, 'cacheV7_train.npz'))
ph2 = np.load(os.path.join(RA, 'autonomous/ra-ph2/cachePH2_train.npz'))
N = len(v7['label'])
assert (pt1['label'][:N] == v7['label']).all() and (ph2['label'][:N] == v7['label']).all()
mel = [pt1['mel']]
lab = [pt1['label']]; src = [pt1['source']]
X = v7['mel'].astype(np.float32)
warped = np.empty_like(X)
alphas = [1.15, 1.25, 1.35]
for k, a in enumerate(alphas):
    idx = np.arange(k, N, 3)
    W = warp_matrix(a)
    warped[idx] = np.einsum('oi,bit->bot', W, X[idx])
mel.append(warped.astype(np.float16)); lab.append(v7['label']); src.append(v7['source'])
mel.append(ph2['mel'][N:]); lab.append(ph2['label'][N:]); src.append(ph2['source'][N:])
M = np.concatenate(mel); L = np.concatenate(lab); S = np.concatenate(src)
np.savez_compressed(os.path.join(HERE, 'cacheC1_train.npz'), mel=M, label=L, source=S)
print('cacheC1', M.shape, 'rows: pt1', len(pt1['label']), '+ vtlp', N, '+ phrase', N)
print('BUILD_C1_DONE')

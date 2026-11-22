"""RA-J1 cache: RA-C1 union cache + retired dev clips x20 (source 'jibo')."""
import os
import numpy as np
RA = os.path.expanduser('~/workspace/micro-edge-pilot/read-aloud')
HERE = os.path.join(RA, 'autonomous/ra-j1')
c1 = np.load(os.path.join(RA, 'autonomous/ra-c1/cacheC1_train.npz'))
dev = np.load(os.path.join(RA, 'child_eval/dev_mel.npz'))
mel = [c1['mel']]; lab = [c1['label']]; src = [c1['source']]
REP = 20
mel.append(np.repeat(dev['mel'], REP, axis=0))
lab.append(np.repeat(dev['y'], REP))
src.append(np.array(['jibo'] * (len(dev['y']) * REP)))
M = np.concatenate(mel); L = np.concatenate(lab); S = np.concatenate(src)
np.savez_compressed(os.path.join(HERE, 'cacheJ1_train.npz'), mel=M, label=L, source=S)
print('cacheJ1', M.shape, '= c1', len(c1['label']), '+ jibo', len(dev['y']) * REP)
print('BUILD_J1_DONE')

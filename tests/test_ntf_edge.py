import numpy as np
import torch
import torchnmf
import h5py

df = h5py.File('tests/example-adjacency.hdf', 'r') 
X = df[[*df.keys()][2]]['data'][...]
X[:, np.arange(X.shape[-1]), np.arange(X.shape[-1])] = 0
nN = X.shape[-1]

X = X[:, *np.triu_indices(X.shape[-1], k=1)]
print(X[0])

X = torch.from_numpy(X)
ntf_mdl = torchnmf.ntf.NTF(X.shape, 3)
print(ntf_mdl.tshape, ntf_mdl().shape)
for mode in range(ntf_mdl.nmodes):
    print(ntf_mdl.modes[mode].shape,
          ntf_mdl.subnmf[mode].W.shape,
          ntf_mdl.subnmf[mode].H.shape)
print(torchnmf.operations.khatri_rao(ntf_mdl.modes[:0] + ntf_mdl.modes[1:]).shape)
ntf_trainer = torchnmf.ntf.NTFTrainer(ntf_mdl, [1.0, 1.0], [1.0, 1.0])

print(np.linalg.norm((X - ntf_mdl()).detach().numpy()))
for i in range(100):
    output = ntf_trainer.model_online_update_and_filter(X, n_iter=1)
print(np.linalg.norm((X - ntf_mdl()).detach().numpy()))


import matplotlib.pyplot as plt
for r in range(ntf_mdl.rank):
    W = ntf_mdl.modes[0].detach()[:, [r]]
    H = ntf_mdl.modes[1].detach()[:, [r]]
    A = np.zeros((nN, nN))
    A[*np.triu_indices(nN, k=1)] = H[:,0]
    A += A.T

    plt.matshow(A)
    plt.title(r)
plt.figure()
plt.plot(ntf_mdl.modes[0].detach())
plt.show()

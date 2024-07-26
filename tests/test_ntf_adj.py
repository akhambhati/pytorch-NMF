import numpy as np
import torch
import torchnmf
import h5py

df = h5py.File('tests/example-adjacency.hdf', 'r') 
X = df[[*df.keys()][2]]['data'][...]
X[:, np.arange(X.shape[-1]), np.arange(X.shape[-1])] = 0
print(X[0])

X = torch.from_numpy(X)
ntf_mdl = torchnmf.netntf.NetNTF(X.shape, 5, shared_modes=[[1,2]])
print(ntf_mdl.tshape, ntf_mdl().shape)
for mode in range(ntf_mdl.nmodes):
    print(ntf_mdl.modes[mode].shape,
          ntf_mdl.subnmf[mode].W.shape,
          ntf_mdl.subnmf[mode].H.shape)
print(torchnmf.operations.khatri_rao(ntf_mdl.modes[:0] + ntf_mdl.modes[1:]).shape)
ntf_trainer = torchnmf.netntf.NetNTFTrainer(ntf_mdl, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0])

print(np.linalg.norm((X - ntf_mdl()).detach().numpy()))
for i in range(1000):
    output = ntf_trainer.model_online_update_and_filter(X, n_iter=1)
    #print(i)
print(np.linalg.norm((X - ntf_mdl()).detach().numpy()))


import matplotlib.pyplot as plt
plt.matshow(X.detach().mean(axis=0));
for r in range(ntf_mdl.rank):
    W = ntf_mdl.modes[1].detach()[:, [r]]
    H = ntf_mdl.modes[2].detach()[:, [r]]
    plt.matshow(W @ H.T)
    plt.title(r)
plt.figure()
plt.plot(ntf_mdl.modes[0].detach())

plt.show()

print(ntf_mdl.modes[1].detach() - ntf_mdl.modes[2].detach())
print(ntf_trainer.trainable_modes)

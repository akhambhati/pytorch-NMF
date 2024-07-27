import numpy as np
import scipy.stats as sp_stats
import torch
import torchnmf
import h5py

df = h5py.File('tests/example-adjacency.hdf', 'r') 
X = df[[*df.keys()][2]]['data'][...]
X[:, np.arange(X.shape[-1]), np.arange(X.shape[-1])] = 0
nN = X.shape[-1]

Xa = X[:, *np.triu_indices(X.shape[-1], k=1)]
Ya = np.random.uniform(size=(X.shape[0], 10))

#Xa = (Xa.T / Xa.sum(axis=-1)).T
#Ya = (Ya.T / Ya.sum(axis=-1)).T

X = torch.from_numpy(Xa)
Y = torch.from_numpy(Ya)
print(X.shape, Y.shape)

ntf_mdl = torchnmf.ssnetntf.SSNetNTF(X.shape, Y.shape, 3, 0, [[0,0]])
net_trainer = torchnmf.ntf.NTFTrainer(ntf_mdl.submdl_net, [1.0, 1.0], [2.0, 2.0])
beh_trainer = torchnmf.ntf.NTFTrainer(ntf_mdl.submdl_beh, [1.0, 1.0], [2.0, 2.0])
ntf_trainer = torchnmf.ssnetntf.SSNetNTFTrainer(ntf_mdl, [net_trainer, beh_trainer])

for m in ntf_mdl.submdl_net.modes:
    print(m.shape)

for m in ntf_mdl.submdl_beh.modes:
    print(m.shape)


print(sp_stats.pearsonr(X.detach().numpy().reshape(-1), ntf_mdl()[0].detach().numpy().reshape(-1)))
print(sp_stats.pearsonr(Y.detach().numpy().reshape(-1), ntf_mdl()[1].detach().numpy().reshape(-1)))
output = ntf_trainer.model_online_update_and_filter(X, Y, n_iter=1000)
print(sp_stats.pearsonr(X.detach().numpy().reshape(-1), ntf_mdl()[0].detach().numpy().reshape(-1)))
print(sp_stats.pearsonr(Y.detach().numpy().reshape(-1), ntf_mdl()[1].detach().numpy().reshape(-1)))


import matplotlib.pyplot as plt

res1 = X.detach().numpy().reshape(-1) / ntf_mdl()[0].detach().numpy().reshape(-1)
dfit1 = sp_stats.fit(sp_stats.expon, res1)
#plt.hist(res1, 50, density=True)
dfit1.plot()
plt.show()

res1 = Y.detach().numpy().reshape(-1) / ntf_mdl()[1].detach().numpy().reshape(-1)
dfit1 = sp_stats.fit(sp_stats.expon, res1)
#plt.hist(res1, 50, density=True)
dfit1.plot()
plt.show()

for r in range(ntf_mdl.rank):
    H = ntf_mdl.submdl_net.modes[1].detach()[:, [r]]
    A = np.zeros((nN, nN))
    A[*np.triu_indices(nN, k=1)] = H[:,0]
    A += A.T

    plt.matshow(A)
    plt.title(r)
plt.show()

H = ntf_mdl.submdl_beh.modes[1].detach()[:, :]
plt.matshow(H, aspect=H.shape[1]/H.shape[0])

W1 = ntf_mdl.submdl_net.modes[0].detach()[:, :]
W2 = ntf_mdl.submdl_beh.modes[0].detach()[:, :]
plt.figure()
plt.plot(W1)

plt.figure()
plt.plot(W2)
plt.show()

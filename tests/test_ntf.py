import numpy as np
import torch
import torchnmf


X = torch.rand(50, 15, 15)
ntf_mdl = torchnmf.ntf.NTF(X.shape, 2)
print(ntf_mdl.tshape, ntf_mdl().shape)
for mode in range(ntf_mdl.nmodes):
    print(ntf_mdl.modes[mode].shape,
          ntf_mdl.subnmf[mode].W.shape,
          ntf_mdl.subnmf[mode].H.shape)
print(torchnmf.operations.khatri_rao(ntf_mdl.modes[:0] + ntf_mdl.modes[1:]).shape)
ntf_trainer = torchnmf.ntf.NTFTrainer(ntf_mdl, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0])

print(np.linalg.norm((X - ntf_mdl()).detach().numpy()))
output = ntf_trainer.model_online_update_and_filter(X, n_iter=1000)
print(np.linalg.norm((X - ntf_mdl()).detach().numpy()))

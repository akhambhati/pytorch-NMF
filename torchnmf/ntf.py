import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nmf import NMF
from .trainer import AdaptiveMu
from .operations import unfold, outer_prod, khatri_rao


class NTF(nn.Module):
    def __init__(self,
            tensor_shape,
            rank):

        super().__init__()
        self.tshape = tensor_shape
        self.nmodes = len(tensor_shape)
        self.rank = rank

        with torch.no_grad():
            self.subnmf = []
            self.modes = []
            for m in range(self.nmodes):
                n1, n2 = self.tshape[m], np.prod(self.tshape[:m] + self.tshape[m+1:]) 
                self.subnmf.append(NMF((n2, n1), rank=self.rank))
                self.modes.append(self.subnmf[-1].W)

    def forward(self):
        return outer_prod(self.modes).sum(axis=-1)

    def subnmf_loss(self, mode, X, beta):
        X_unfold = unfold(X, mode)
        Xh_unfold = unfold(self.forward(), mode)

        io_dict = {}
        for pn, p in self.subnmf[mode].named_parameters():
            if id(p) not in io_dict:
                io_dict[id(p)] = list()
            penalty = torch.zeros_like(p)
            io_dict[id(p)].append((X_unfold, Xh_unfold, beta, penalty))

        return io_dict

    def update_subnmf_kr(self):
        for m in range(self.nmodes): 
            kr = khatri_rao(self.modes[:m] + self.modes[m+1:])
            self.subnmf[m].H[...] = kr
             
    def reinit_mode(self, m):
        self.subnmf[m].W = torch.rand(self.subnmf[m].W.shape).abs() 


class NTFTrainer():
    def __init__(self,
            ntf_model,
            modes_lr,
            modes_beta):

        self.ntf_model = ntf_model
        self.modes_lr = [lr*np.ones((1, ntf_model.rank)) for lr in modes_lr]
        self.modes_beta = modes_beta

        self.subnmf_trainers = [AdaptiveMu(
            params=[self.ntf_model.subnmf[m].W],
            theta=[self.modes_lr[m]]) for m in range(ntf_model.nmodes)]

    def train_W(self, signal, mode, reinit=True, n_iter=1):
        with torch.no_grad():
            if reinit:
                self.ntf_model.reinit_mode(mode)

        for i in range(n_iter):
            def closure():
                self.subnmf_trainers[mode].zero_grad()
                return self.ntf_model.subnmf_loss(
                        mode,
                        signal,
                        self.modes_beta[mode])
            self.subnmf_trainers[mode].step(closure)

    def model_online_update_and_filter(self, signal, n_iter):
        for mode in range(self.ntf_model.nmodes):
            with torch.no_grad():
                self.ntf_model.update_subnmf_kr()
            self.train_W(signal, mode, reinit=False, n_iter=n_iter)
        
        return self

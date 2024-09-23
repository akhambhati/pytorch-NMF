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
            rank, constraints=None):

        super().__init__()
        self.tshape = tensor_shape
        self.nmodes = len(tensor_shape)
        self.rank = rank

        self.lam = torch.ones(self.rank)
        with torch.no_grad():
            self.subnmf = []
            self.modes = []
            for m in range(self.nmodes):
                n1, n2 = self.tshape[m], np.prod(self.tshape[:m] + self.tshape[m+1:]) 
                self.subnmf.append(NMF((n2, n1), rank=self.rank))
                self.modes.append(self.subnmf[-1].W)
                self.reinit_mode(m)
                self.norm_mode_within_factor(m)

        if constraints is None:
            constraints = {}

        for mode in range(self.nmodes):
            if mode not in constraints:
                constraints[mode] = {}
        self.constraints = constraints

    def forward(self):
        return outer_prod(self.modes).sum(axis=-1)

    def subnmf_loss(self, mode, X, beta, M=None, io_dict=None):
        X_unfold = unfold(X, mode)
        Xh_unfold = unfold(self.forward(), mode)
        if M is None:
            M = torch.ones_like(X)
        M_unfold = unfold(M, mode)

        if io_dict is None:
            io_dict = {}

        for pn, p in self.subnmf[mode].named_parameters():
            if id(p) not in io_dict:
                io_dict[id(p)] = list()
            penalty = self.calc_penalty(mode)
            io_dict[id(p)].append((X_unfold, Xh_unfold, beta, penalty, M_unfold))
        return io_dict

    def calc_penalty(self, mode):
        pen = torch.zeros_like(self.subnmf[mode].W) 

        if ('l1' in self.constraints[mode]):
            for c in self.constraints[mode]['l1']:
                assert c.shape == pen.shape
                pen += c

        if ('ortho' in self.constraints[mode]):
            for c in self.constraints[mode]['ortho']:
                pen += (c * (torch.ones((self.rank, self.rank)) - torch.eye(self.rank)) @ self.subnmf[mode].W.detach().T).T

        if ('xortho' in self.constraints[mode]):
            for c in self.constraints[mode]['xortho']:
                if c[2] is None:
                    c[2] = (torch.ones((self.rank, self.rank)) - torch.eye(self.rank))
                pen += (c[0] * (unfold(c[1], mode) @ self.subnmf[mode].H.detach()) @ c[2])
        return pen

    def update_subnmf_kr(self):
        for m in range(self.nmodes): 
            kr = khatri_rao(self.modes[:m] + self.modes[m+1:])
            self.subnmf[m].H[...] = kr

    def reinit_mode(self, m):
        self.subnmf[m].W[...] = torch.rand(self.subnmf[m].W.shape).abs() 

    def norm_mode_within_factor(self, mode):
        tot_pow = self.subnmf[mode].W[:, :].sum(axis=0)
        tot_pow = torch.nan_to_num(tot_pow, nan=1.0)
        self.subnmf[mode].W[:, :] = (
                self.subnmf[mode].W[:, :] / tot_pow)
        self.lam *= tot_pow

    def norm_mode_across_factor(self, mode):
        tot_pow = self.subnmf[mode].W[:, :].sum(axis=1)
        tot_pow = torch.nan_to_num(tot_pow, nan=1.0)
        self.subnmf[mode].W[:, :] = (
                self.subnmf[mode].W[:, :].T / tot_pow).T


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
        self.trainable_modes = [*range(self.ntf_model.nmodes)] 

    def train_W(self, signal, mode, mask=None, reinit=True, n_iter=1):
        with torch.no_grad():
            if reinit:
                self.ntf_model.reinit_mode(mode)

        for i in range(n_iter):
            def closure():
                self.subnmf_trainers[mode].zero_grad()
                return self.ntf_model.subnmf_loss(
                        mode,
                        signal,
                        self.modes_beta[mode],
                        mask)
            self.subnmf_trainers[mode].step(closure)

    def model_online_update_and_filter(self, signal, n_iter,
            reinit_modes=[],
            norm_modes_within_factor=[],
            norm_modes_across_factor=[],
            mask=None):
        for mode in self.trainable_modes:
            with torch.no_grad():
                self.ntf_model.update_subnmf_kr()
            self.train_W(signal, mode, mask, reinit=True if mode in reinit_modes else False,
                    n_iter=n_iter)

        with torch.no_grad():
            for mode in norm_modes_within_factor:
                self.ntf_model.norm_mode_within_factor(mode)
            for mode in norm_modes_across_factor:
                self.ntf_model.norm_mode_across_factor(mode)

        return self

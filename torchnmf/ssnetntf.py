import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ntf import NTF
from .netntf import NetNTF
from .trainer import AdaptiveMu
from .operations import unfold, outer_prod, khatri_rao


class SSNetNTF(nn.Module):
    def __init__(self,
            network_shape,
            behavior_shape,
            rank_b, rank_nb,
            shared_modes=[],
            fit_intercept=False,
            shared_norm=True):

        super().__init__()
        self.fit_intercept = fit_intercept
        if self.fit_intercept:
            self.rank_b = rank_b + 1
        else:
            self.rank_b = rank_b
        self.rank_nb = rank_nb
        self.rank = self.rank_b + self.rank_nb
        self.shared_modes = shared_modes
        self.shared_norm = shared_norm

        self.submdl_net = NTF(network_shape, self.rank) 
        self.submdl_beh = NTF(behavior_shape, self.rank)

        for shared in self.shared_modes:
            assert self.submdl_net.modes[shared[0]].shape == self.submdl_beh.modes[shared[1]].shape
            self.submdl_net.subnmf[shared[0]].W = self.submdl_beh.subnmf[shared[1]].W
            self.submdl_net.modes[shared[0]] = self.submdl_beh.modes[shared[1]]

        with torch.no_grad():
            self.zero_submdl_beh()
            if self.fit_intercept:
                self.fix_intercept()
            if self.shared_norm:
                self.norm_shared_mode()

    def forward(self):
        return self.submdl_net.forward(), self.submdl_beh.forward()

    def zero_submdl_beh(self):
        all_shared_modes = [shared[1] for shared in self.shared_modes]
        for r in range(self.rank_b, self.rank):
            for m in range(len(self.submdl_beh.modes)):
                if m in all_shared_modes:
                    continue
                self.submdl_beh.subnmf[m].W[:, r] = 0

    def fix_intercept(self):
        for shared in self.shared_modes:
            self.submdl_net.subnmf[shared[0]].W[:, 0] = 1
            self.submdl_beh.subnmf[shared[1]].W[:, 0] = 1

    def norm_shared_mode(self):
        for shared in self.shared_modes:
            tot_pow = self.submdl_net.subnmf[shared[0]].W[:, :].sum(axis=1)
            self.submdl_net.subnmf[shared[0]].W[:, :] = (
                    self.submdl_net.subnmf[shared[0]].W[:, :].T / tot_pow).T
            tot_pow = self.submdl_beh.subnmf[shared[1]].W[:, :].sum(axis=1)
            self.submdl_beh.subnmf[shared[1]].W[:, :] = (
                    self.submdl_beh.subnmf[shared[1]].W[:, :].T / tot_pow).T


class SSNetNTFTrainer():
    def __init__(self,
            ssnetntf_model,
            ntf_trainers):

        self.ntf_model = ssnetntf_model
        self.submdl_trainers = ntf_trainers

    def model_online_update_and_filter(self, signal_net, signal_beh, n_iter):
        for i in range(n_iter):
            with torch.no_grad():
                self.ntf_model.zero_submdl_beh()
                if self.ntf_model.fit_intercept:
                    self.ntf_model.fix_intercept()
                if self.ntf_model.shared_norm:
                    self.ntf_model.norm_shared_mode()
                self.ntf_model.submdl_net.update_subnmf_kr()
                self.ntf_model.submdl_beh.update_subnmf_kr()

            # Shared modes first
            sel_mode = self.ntf_model.shared_modes[0][0] 
            io_dict = self.ntf_model.submdl_net.subnmf_loss(
                    sel_mode, signal_net,
                    self.submdl_trainers[0].modes_beta[sel_mode])

            sel_mode = self.ntf_model.shared_modes[0][1]
            io_dict = self.ntf_model.submdl_beh.subnmf_loss(
                    sel_mode, signal_beh,
                    self.submdl_trainers[1].modes_beta[sel_mode],
                    io_dict)

            self.submdl_trainers[0].subnmf_trainers[self.ntf_model.shared_modes[0][0]].step(lambda: io_dict)

            # Individual modes
            for sel_mode in self.submdl_trainers[0].trainable_modes:
                if sel_mode == self.ntf_model.shared_modes[0][0]:
                    continue
                with torch.no_grad():
                    self.ntf_model.zero_submdl_beh()
                    if self.ntf_model.fit_intercept:
                        self.ntf_model.fix_intercept()
                    if self.ntf_model.shared_norm:
                        self.ntf_model.norm_shared_mode()
                    self.ntf_model.submdl_net.update_subnmf_kr()

                io_dict = self.ntf_model.submdl_net.subnmf_loss(
                        sel_mode, signal_net,
                        self.submdl_trainers[0].modes_beta[sel_mode])
                self.submdl_trainers[0].subnmf_trainers[sel_mode].step(lambda: io_dict)

            for sel_mode in self.submdl_trainers[1].trainable_modes:
                if sel_mode == self.ntf_model.shared_modes[0][1]:
                    continue
                with torch.no_grad():
                    self.ntf_model.zero_submdl_beh()
                    if self.ntf_model.fit_intercept:
                        self.ntf_model.fix_intercept()
                    if self.ntf_model.shared_norm:
                        self.ntf_model.norm_shared_mode()
                    self.ntf_model.submdl_beh.update_subnmf_kr()

                io_dict = self.ntf_model.submdl_beh.subnmf_loss(
                        sel_mode, signal_beh,
                        self.submdl_trainers[1].modes_beta[sel_mode])
                self.submdl_trainers[1].subnmf_trainers[sel_mode].step(lambda: io_dict)  

        return self

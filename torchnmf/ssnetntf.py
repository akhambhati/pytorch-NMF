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
            shared_modes=[]):

        super().__init__()
        self.rank_b = rank_b
        self.rank_nb = rank_nb
        self.rank = self.rank_b + self.rank_nb
        self.shared_modes = shared_modes

        self.submdl_net = NTF(network_shape, self.rank) 
        self.submdl_beh = NTF(behavior_shape, self.rank)

        for shared in self.shared_modes:
            assert self.submdl_net.modes[shared[0]].shape == self.submdl_beh.modes[shared[1]].shape
            self.submdl_net.subnmf[shared[0]].W = self.submdl_beh.subnmf[shared[1]].W
            self.submdl_net.modes[shared[0]] = self.submdl_beh.modes[shared[1]]

    def forward(self):
        return self.submdl_net.forward(), self.submdl_beh.forward()

    def forward_net(self):
        return self.submdl_net.forward()

    def forward_beh(self):
        return self.submdl_beh.forward()

    def zero_nb_factors(self):
        all_shared_modes = [shared[1] for shared in self.shared_modes]
        for r in range(self.rank_b, self.rank):
            for m in range(len(self.submdl_beh.modes)):
                if m in all_shared_modes:
                    continue
                self.submdl_beh.subnmf[m].W[:, r] = 0


class SSNetNTFTrainer():
    def __init__(self,
            ssnetntf_model,
            ntf_trainers):

        self.ntf_model = ssnetntf_model
        self.submdl_trainers = ntf_trainers

    def filter_net_and_beh(self, signal_net, signal_beh, mask_net, mask_beh, n_iter, reinit_mode=False, norm_factor=None):
        with torch.no_grad():
            if reinit_mode:
                self.ntf_model.submdl_net.reinit_mode(
                        self.ntf_model.shared_modes[0][0])
                self.ntf_model.submdl_beh.reinit_mode(
                        self.ntf_model.shared_modes[0][1])
            self.ntf_model.submdl_net.update_subnmf_kr()
            self.ntf_model.submdl_beh.update_subnmf_kr()

        io_dict = None

        sel_mode = self.ntf_model.shared_modes[0][0]
        if sel_mode in self.submdl_trainers[0].trainable_modes:
            io_dict = self.ntf_model.submdl_net.subnmf_loss(
                    sel_mode, signal_net,
                    self.submdl_trainers[0].modes_beta[sel_mode],
                    mask_net,
                    io_dict)

        sel_mode = self.ntf_model.shared_modes[0][1]
        if sel_mode in self.submdl_trainers[1].trainable_modes:
            io_dict = self.ntf_model.submdl_beh.subnmf_loss(
                    sel_mode, signal_beh,
                    self.submdl_trainers[1].modes_beta[sel_mode],
                    mask_beh,
                    io_dict)

        self.submdl_trainers[0].subnmf_trainers[self.ntf_model.shared_modes[0][0]].step(lambda: io_dict)

        with torch.no_grad():
            if norm_factor == 'within':
                sel_mode = self.ntf_model.shared_modes[0][0]
                self.ntf_model.submdl_net.norm_mode_within_factor(sel_mode)

                sel_mode = self.ntf_model.shared_modes[0][1]
                self.ntf_model.submdl_beh.norm_mode_within_factor(sel_mode)

            elif norm_factor == 'across':
                sel_mode = self.ntf_model.shared_modes[0][0]
                self.ntf_model.submdl_net.norm_mode_across_factor(sel_mode)

                sel_mode = self.ntf_model.shared_modes[0][1]
                self.ntf_model.submdl_beh.norm_mode_across_factor(sel_mode)
            else:
                pass

    def filter_net(self, signal_net, mask_net, n_iter, reinit_mode=False, norm_factor=None):
        with torch.no_grad():
            if reinit_mode:
                self.ntf_model.submdl_net.reinit_mode(
                        self.ntf_model.shared_modes[0][0])
            self.ntf_model.submdl_net.update_subnmf_kr()

        io_dict = None

        sel_mode = self.ntf_model.shared_modes[0][0]
        if sel_mode in self.submdl_trainers[0].trainable_modes:
            io_dict = self.ntf_model.submdl_net.subnmf_loss(
                    sel_mode, signal_net,
                    self.submdl_trainers[0].modes_beta[sel_mode],
                    mask_net,
                    io_dict)

        self.submdl_trainers[0].subnmf_trainers[self.ntf_model.shared_modes[0][0]].step(lambda: io_dict)

        with torch.no_grad():
            if norm_factor == 'within':
                sel_mode = self.ntf_model.shared_modes[0][0]
                self.ntf_model.submdl_net.norm_mode_within_factor(sel_mode)

            elif norm_factor == 'across':
                sel_mode = self.ntf_model.shared_modes[0][0]
                self.ntf_model.submdl_net.norm_mode_across_factor(sel_mode)

            else:
                pass

    def filter_beh(self, signal_beh, mask_beh, n_iter, reinit_mode=False, norm_factor=None):
        with torch.no_grad():
            if reinit_mode:
                self.ntf_model.submdl_beh.reinit_mode(
                        self.ntf_model.shared_modes[0][1])
            self.ntf_model.submdl_beh.update_subnmf_kr()

        io_dict = None

        sel_mode = self.ntf_model.shared_modes[0][1]
        if sel_mode in self.submdl_trainers[1].trainable_modes:
            io_dict = self.ntf_model.submdl_beh.subnmf_loss(
                    sel_mode, signal_beh,
                    self.submdl_trainers[1].modes_beta[sel_mode],
                    mask_beh,
                    io_dict)

        self.submdl_trainers[1].subnmf_trainers[self.ntf_model.shared_modes[0][1]].step(lambda: io_dict)

        with torch.no_grad():
            if norm_factor == 'within':
                sel_mode = self.ntf_model.shared_modes[0][1]
                self.ntf_model.submdl_beh.norm_mode_within_factor(sel_mode)

            elif norm_factor == 'across':
                sel_mode = self.ntf_model.shared_modes[0][1]
                self.ntf_model.submdl_beh.norm_mode_across_factor(sel_mode)
            else:
                pass

    def update_net(self, signal_net, mask_net, n_iter, reinit_modes=[], norm_factor=None):
        # Individual modes
        for sel_mode in self.submdl_trainers[0].trainable_modes:
            if sel_mode == self.ntf_model.shared_modes[0][0]:
                continue
            with torch.no_grad():
                if sel_mode in reinit_modes:
                    self.ntf_model.submdl_net.reinit_mode(
                            sel_mode)
                self.ntf_model.zero_nb_factors()
                self.ntf_model.submdl_net.update_subnmf_kr()

            io_dict = self.ntf_model.submdl_net.subnmf_loss(
                    sel_mode, signal_net,
                    self.submdl_trainers[0].modes_beta[sel_mode], mask_net)
            self.submdl_trainers[0].subnmf_trainers[sel_mode].step(lambda: io_dict)

            with torch.no_grad():
                if norm_factor == 'within':
                    self.ntf_model.submdl_net.norm_mode_within_factor(sel_mode)
                elif norm_factor == 'across':
                    self.ntf_model.submdl_net.norm_mode_across_factor(sel_mode)
                else:
                    pass

            with torch.no_grad():
                self.ntf_model.zero_nb_factors()

    def update_beh(self, signal_beh, mask_beh, n_iter, reinit_modes=[], norm_factor=None):
        # Individual modes
        for sel_mode in self.submdl_trainers[1].trainable_modes:
            if sel_mode == self.ntf_model.shared_modes[0][1]:
                continue
            with torch.no_grad():
                if sel_mode in reinit_modes:
                    self.ntf_model.submdl_beh.reinit_mode(
                            sel_mode)
                self.ntf_model.zero_nb_factors()
                self.ntf_model.submdl_beh.update_subnmf_kr()

            io_dict = self.ntf_model.submdl_beh.subnmf_loss(
                    sel_mode, signal_beh,
                    self.submdl_trainers[1].modes_beta[sel_mode], mask_beh)
            self.submdl_trainers[1].subnmf_trainers[sel_mode].step(lambda: io_dict)

            with torch.no_grad():
                if norm_factor == 'within':
                    self.ntf_model.submdl_beh.norm_mode_within_factor(sel_mode)
                elif norm_factor == 'across':
                    self.ntf_model.submdl_beh.norm_mode_across_factor(sel_mode)
                else:
                    pass

            with torch.no_grad():
                self.ntf_model.zero_nb_factors()

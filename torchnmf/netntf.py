import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ntf import NTF, NTFTrainer
from .trainer import AdaptiveMu
from .operations import unfold, outer_prod, khatri_rao


class NetNTF(NTF):
    def __init__(self,
            tensor_shape,
            rank,
            shared_modes=[]):

        super().__init__(tensor_shape, rank)
        self.shared_modes = shared_modes
        for shared in self.shared_modes:
            a = self.subnmf[shared[0]] 
            for smode in shared:
                self.subnmf[smode] = a
            a = self.modes[shared[0]] 
            for smode in shared:
                self.modes[smode] = a


class NetNTFTrainer(NTFTrainer):
    def __init__(self,
            ntf_model,
            modes_lr,
            modes_beta):
        super().__init__(ntf_model, modes_lr, modes_beta)
        for shared in self.ntf_model.shared_modes:
            for smode in shared[:-1]:
                self.trainable_modes.remove(smode)

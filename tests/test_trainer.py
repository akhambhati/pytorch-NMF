import pytest
import torch
from torch import nn

from torchnmf.trainer import *
from torchnmf.nmf import NMF
from torchnmf.metrics import beta_div


@pytest.mark.parametrize('beta', [-1, 0, 0.5, 1, 1.5, 2, 3])
@pytest.mark.parametrize('alpha', [0, 1e-3])
@pytest.mark.parametrize('l1_ratio', [0, 0.5, 1])
@pytest.mark.parametrize('theta', [0, 0.5, 1])
def test_beta_trainer(beta, alpha, l1_ratio, theta):
    m = nn.Sequential(
        NMF((100, 16), rank=8),
        NMF(W=(32, 16)),
        NMF(W=(50, 32))
    )

    target = torch.rand(100, 50)
    trainer = AdaptiveMu(m.parameters(), beta, alpha, l1_ratio, theta)

    def closure():
        trainer.zero_grad()
        return target, m(None)

    for _ in range(10):
        trainer.step(closure)
        for p in m.parameters():
            assert torch.all(p >= 0.)
    return

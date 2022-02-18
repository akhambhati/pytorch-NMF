import pytest
import torch


from torchnmf.nmf import *


@pytest.mark.parametrize('rank', [8])
@pytest.mark.parametrize('W', [(50, 8), torch.rand(50, 8), None])
@pytest.mark.parametrize('H', [(100, 8), torch.rand(100, 8), None])
def test_base_valid_construct(rank, W, H):
    m = BaseComponent(rank, W, H)
    if H is None:
        assert m.H is None
    if W is None:
        assert m.W is None


@pytest.mark.parametrize('rank, W, H',
                         [(None, None, None),
                          (None, (50, 8), (100, 10)),
                          (None, torch.rand(50, 8), (100, 10)),
                          (None, torch.randn(50, 8), (100, 8)),
                          (None, (50, 8), torch.rand(100, 10)),
                          (None, (50, 8), torch.randn(100, 8)),
                          (None, torch.rand(50, 8), torch.rand(100, 10)),
                          (None, torch.randn(50, 8), torch.rand(100, 8)),
                          (None, torch.rand(50, 8), torch.randn(100, 8)),
                          (None, torch.randn(50, 8), torch.randn(100, 8)), ]
                         )
def test_base_invalid_construct(rank, W, H):
    try:
        m = BaseComponent(rank, W, H)
    except:
        assert True
    else:
        assert False, "Should not reach here"


def test_nmf_valid_construct():
    m = NMF((100, 50))
    y = m()
    assert y.shape == (100, 50)


@pytest.mark.parametrize('Vshape', [(100, 50, 50), (100,)])
def test_nmf_invalid_construct(Vshape):
    try:
        m = NMF(Vshape)
    except:
        assert True
    else:
        assert False, "Should not reach here"


def test_nmfd_valid_construct():
    m = NMFD((100, 50, 100))
    y = m()
    assert y.shape == (100, 50, 100)


@pytest.mark.parametrize('Vshape', [(100, 50), (100,), (100, 50) * 2])
def test_nmfd_invalid_construct(Vshape):
    try:
        m = NMFD(Vshape)
    except:
        assert True
    else:
        assert False, "Should not reach here"


def test_nmf2d_valid_construct():
    m = NMF2D((8, 32, 100, 100), 16)
    y = m()
    assert y.shape == (8, 32, 100, 100)


@pytest.mark.parametrize('Vshape', [(100, 50), (100,), (100, 50) * 6])
def test_nmf2d_invalid_construct(Vshape):
    try:
        m = NMF2D(Vshape)
    except:
        assert True
    else:
        assert False, "Should not reach here"


def test_nmf3d_valid_construct():
    m = NMF3D((8, 50, 100, 100, 100), 8)
    y = m()
    assert y.shape == (8, 50, 100, 100, 100)


@pytest.mark.parametrize('Vshape', [(100, 50), (100,), (100, 50) * 4])
def test_nmf3d_invalid_construct(Vshape):
    try:
        m = NMF3D(Vshape)
    except:
        assert True
    else:
        assert False, "Should not reach here"

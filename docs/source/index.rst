:github_url: https://github.com/yoyololicon/pytorch-NMF

PyTorch NMF Documentation
===============================

PyTorch NMF is a extension library for `PyTorch <https://pytorch.org/>`_.

It consists of basic NMF algorithm and its convolutional variants, which are hardly found in other NMF packages.
In addition, by using the PyTorch automatic differentiation feature, it is able to adopt the classic multiplicative update rules
into more complex NMF structures, and make it possible to train these complex models in a simple end-to-end fashion.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/intro

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/nmf
   modules/plca
   modules/metrics
   modules/trainer


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

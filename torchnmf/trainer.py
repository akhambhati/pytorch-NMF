import torch
from torch.optim.optimizer import Optimizer, required
from .constants import eps

class AdaptiveMu(Optimizer):
    r"""Implements the classic multiplicative updater for NMF models minimizing β-divergence.

    Note:
        To use this optimizer, not only make sure your model parameters are non-negative, but the gradients
        along the whole computational graph are always non-negative.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        beta (float, optional): beta divergence to be minimized, measuring the distance between target and the NMF model.
                        Default: ``1.``
        alpha (float, optional): Weight of the elastic net regularizer. Default: ``0.0``.
        l1_ratio (float, optional): Relative L1/L2 regularization for elastic net. Default: ``0.5``.
        theta (float, optional): coefficient used for weighing relative 
            contribution of past gradient and current gradient. (Default: ``(1.0)``)
    """

    def __init__(self, params, beta=1, alpha=0, l1_ratio=0.5, theta=1):
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= l1_ratio <= 1:
            raise ValueError("Invalid l1_ratio value: {}".format(l1_ratio))
        if not 0.0 <= theta <= 1.0:
            raise ValueError("Invalid theta parameter value: {}".format(theta))

        defaults = dict(
                beta=beta,
                alpha=alpha,
                l1_ratio=l1_ratio,
                theta=theta)
        super(AdaptiveMu, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        """Performs a single update step.

        Arguments:
            closure (callable): a closure that reevaluates the model
                and returns the target and predicted Tensor in the form:
                ``func()->Tuple(target,predict)``
        """

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        # Cache the gradient status for reversion at the end of the function
        status_cache = dict()
        for group in self.param_groups:
            for p in group['params']:
                status_cache[id(p)] = p.requires_grad
                p.requires_grad = False

        # Iterate over each parameter group (specifies order of optimization)
        for group in self.param_groups:
            beta = group['beta']
            alpha = group['alpha']
            l1_ratio = group['l1_ratio']
            theta = group['theta']

            # Iterate over model parameters within the group
            # if a gradient is not required then that parameter is "fixed"
            for p in group['params']:
                if not status_cache[id(p)]:
                    continue
                p.requires_grad = True

                # Close the optimization loop by retrieving the
                # observed data and prediction
                V, WH = closure()
                if not WH.requires_grad:
                    p.requires_grad = False
                    continue

                # Initialize the state variables for gradient averaging.
                # Enables incremental learning.
                # TODO: Lazy state initialization, should init in constructor
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['neg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['pos'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                neg, pos = state['neg'], state['pos']
                state['step'] += 1

                # Multiplicative update coefficients for beta-divergence
                #      Marmin, A., Goulart, J.H.D.M. and Févotte, C., 2021.
                #      Joint Majorization-Minimization for Nonnegative Matrix
                #      Factorization with the $\beta $-divergence. 
                #      arXiv preprint arXiv:2106.15214.
                if beta == 2:
                    output_neg = V
                    output_pos = WH
                elif beta == 1:
                    output_neg = V / WH.add(eps)
                    output_pos = torch.ones_like(WH)
                elif beta == 0:
                    WH_eps = WH.add(eps)
                    output_pos = WH_eps.reciprocal_()
                    output_neg = output_pos.square().mul_(V)
                else:
                    WH_eps = WH.add(eps)
                    output_neg = WH_eps.pow(beta - 2).mul_(V)
                    output_pos = WH_eps.pow_(beta - 1)

                # Numerator (negative factor) gradient
                WH.backward(output_neg, retain_graph=True)
                _neg = torch.clone(p.grad).relu_()
                p.grad.zero_()

                # Denominator (positive factor) gradient
                WH.backward(output_pos)
                _pos = torch.clone(p.grad).relu_()
                p.grad.zero_()

                # Add elastic_net regularizers to the denominator factor
                _pos.add_(alpha*(l1_ratio))
                _pos.add_(p, alpha=(alpha*(1-l1_ratio)))

                # Accumulate gradients 
                neg.mul_(1-theta).add_(_neg.mul_(theta))
                pos.mul_(1-theta).add_(_pos.mul_(theta))

                # Avoid ill-conditioned, zero-valued multipliers
                neg.add_(eps)
                pos.add_(eps)

                # Compute the current state of the parameter
                p.mul_(neg.div(pos))

                p.requires_grad = False

        # Reinstate the grad status from before
        for group in self.param_groups:
            for p in group['params']:
                p.requires_grad = status_cache[id(p)]

        return None

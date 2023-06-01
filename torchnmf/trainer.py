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
        theta (float, optional): coefficient used for weighing relative 
            contribution of past gradient and current gradient. (Default: ``(1.0)``)
    """

    def __init__(self, params, theta):
        if len(params) != len(theta):
            raise ValueError('Length of parameter groups not equal to the length theta')

        for par, th in zip(params, theta):
            if par.shape[1] != th.shape[-1]:
                raise ValueError('Rank of parameters must equal number of learning rate.')
            for t in th:
                if not 0.0 <= t <= 1.0:
                    raise ValueError("Theta should be bounded between 0 and 1")

        defaults = dict(
                theta=theta,
                )
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

        ### FIRST PASS -- Accumulate Positive/Negative Gradient Contributions
        # Iterate over each parameter group (specifies order of optimization)
        for group in self.param_groups:

            # Iterate over model parameters within the group
            # if a gradient is not required then that parameter is "fixed"
            _neg, _pos = None, None
            for p, th in zip(group['params'], group['theta']):
                if not status_cache[id(p)]:
                    continue
                p.requires_grad = True

                # Initialize temporary gradient components
                _neg = torch.zeros_like(p)
                _pos = torch.zeros_like(p)

                # Close the optimization loop by retrieving the
                # observed data and prediction
                loss_fns = closure()[id(p)]
                for loss_fn in loss_fns:
                    with torch.enable_grad():
                        V, WH, beta, penalty = loss_fn
                    if not WH.requires_grad:
                        continue

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
                    # Retain graph so that backward can be run again using the 
                    # positive component.
                    WH.backward(output_neg, retain_graph=True)
                    __neg = (torch.clone(p.grad).relu_())
                    p.grad.zero_()

                    # Denominator (positive factor) gradient
                    # The parameter gradient holds both components (positive - negative)
                    WH.backward(output_pos)
                    __pos = (torch.clone(p.grad).relu_())
                    #p.grad.add_(-_neg)
                    p.grad.zero_()

                    # Include penalty
                    __pos.add_(penalty)

                    # Add to the running estimate
                    _neg.add_(__neg)
                    _pos.add_(__pos)

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

                # Accumulate gradients
                theta = torch.Tensor(th.reshape(1, len(th), 1))
                neg.mul_(1-theta).add_(_neg.mul_(theta))
                pos.mul_(1-theta).add_(_pos.mul_(theta))

                # Avoid ill-conditioned, zero-valued multipliers
                neg.add_(eps)
                pos.add_(eps)

                # Multiplicative Update
                multiplier = neg.div(pos)
                p.mul_(multiplier)

                # Force the gradient requirement to off
                p.requires_grad = False

        # Reinstate the grad status from before
        for group in self.param_groups:
            for p in group['params']:
                p.requires_grad = status_cache[id(p)]

        return None

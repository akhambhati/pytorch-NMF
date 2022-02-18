import torch
from torch.optim.optimizer import Optimizer
from .constants import eps


class BetaMu(Optimizer):
    r"""Implements the classic multiplicative updater for NMF models minimizing β-divergence.

    Note:
        To use this optimizer, not only make sure your model parameters are non-negative, but the gradients
        along the whole computational graph are always non-negative.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        beta (float, optional): beta divergence to be minimized, measuring the distance between target and the NMF model.
                        Default: ``1.``
        l1_reg (float, optional): L1 regularize penalty. Default: ``0.``.
        l2_reg (float, optional): L2 regularize penalty (weight decay). Default: ``0.``
        orthogonal (float, optional): orthogonal regularize penalty. Default: ``0.``
        theta (float, optional): coefficient used for weighing relative 
            contribution of past gradient and current gradient. (Default: ``(1.0)``)
        normalize (bool, optional): perform normalization for each component
            after each iteration.
        update_state (bool, optional): update the historical state based on current
            gradient.
    """

    def __init__(self, params, beta=1, l1_reg=0, l2_reg=0, orthogonal=0, theta=1, normalize=False, update_state=False):
        if not 0.0 <= l1_reg:
            raise ValueError("Invalid l1_reg value: {}".format(l1_reg))
        if not 0.0 <= l2_reg:
            raise ValueError("Invalid l2_reg value: {}".format(l2_reg))
        if not 0.0 <= orthogonal:
            raise ValueError("Invalid orthogonal value: {}".format(orthogonal))
        if not 0.0 <= theta <= 1.0:
            raise ValueError("Invalid theta parameter value: {}".format(theta))
        if not normalize in [True, False]:
            raise ValueError("Invalid normalize parameter value: {}".format(normalize))
        if not update_state in [True, False]:
            raise ValueError("Invalid update_state parameter value: {}".format(normalize))

        defaults = dict(beta=beta, l1_reg=l1_reg,
                        l2_reg=l2_reg, orthogonal=orthogonal, theta=theta,
                        normalize=normalize, update_state=update_state)
        super(BetaMu, self).__init__(params, defaults)

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
            l1_reg = group['l1_reg']
            l2_reg = group['l2_reg']
            ortho = group['orthogonal']
            theta = group['theta']
            normalize = group['normalize']
            update_state = group['update_state']

            # TODO: Warn that if theta < 1 and update_state is False that the
            # gradients will not incorporate previous state information.

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
                neg = torch.clone(p.grad).relu_()
                p.grad.zero_()

                # Denominator (positive factor) gradient
                WH.backward(output_pos)
                pos = torch.clone(p.grad).relu_()
                p.grad.add_(-neg)

                # Add regularizers to the denominator factor
                if l1_reg > 0:
                    pos.add_(l1_reg)
                if l2_reg > 0:
                    pos.add_(p, alpha=l2_reg)
                if ortho > 0:
                    pos.add_(p.sum(1, keepdims=True) - p, alpha=ortho)

                # Cache the multiplicative update numerator/denominator as state
                # variables within the optimizer. Enables incremental learning.
                # TODO: Lazy state initialization, should init in constructor
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 1
                    state['neg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['pos'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Accumulate gradients 
                neg = (1-theta)*state['neg'] + (theta)*neg
                pos = (1-theta)*state['pos'] + (theta)*pos

                # Avoid ill-conditioned, zero-valued multipliers
                pos.add_(eps)
                neg.add_(eps)

                # Compute the current state of the parameter
                p.mul_(neg / pos)

                # If normalize then rescale based on beta-divgerence
                if normalize:

                    # TODO: Iterator should reference a rank parameter rather than
                    # matrix shape.
                    for r in range(p.shape[1]):
                        if beta == 0:
                            norm = (p[:,r] > 0).sum()
                        else:
                            norm = (p[:,r]**beta).sum()**(1/beta)
                        p[:,r] = p[:,r] / norm
                        neg[:, r] = neg[:, r] / norm
                        pos[:, r] = pos[:, r] * norm

                if update_state:
                    state['neg'] = neg
                    state['pos'] = pos
                    state['step'] += 1

                p.requires_grad = False

        # Reinstate the grad status from before
        for group in self.param_groups:
            for p in group['params']:
                p.requires_grad = status_cache[id(p)]

        return None

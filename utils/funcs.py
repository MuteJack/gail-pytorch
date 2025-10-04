# utils/funcs.py

""" Import Library """
# Third-party library imports
import numpy as np
import torch

# Local application imports
from utils.logger import get_logger

# Initialize logger
logger = get_logger()


""" Parameter and Gradient Utilities """
def get_flat_grads(f, net):
    """
    Get flattened gradients of scalar f with respect to network parameters

    Args:
        f: Scalar tensor (loss or objective)
        net: Neural network module

    Returns:
        Flattened gradient vector
    """
    flat_grads = torch.cat([
        grad.view(-1)
        for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
    ])

    return flat_grads


def get_flat_params(net):
    """
    Get flattened parameters of network

    Args:
        net: Neural network module

    Returns:
        Flattened parameter vector
    """
    return torch.cat([param.view(-1) for param in net.parameters()])


def set_params(net, new_flat_params):
    """
    Set network parameters from flattened vector

    Args:
        net: Neural network module
        new_flat_params: Flattened parameter vector
    """
    start_idx = 0
    for param in net.parameters():
        # Calculate end index for this parameter
        end_idx = start_idx + np.prod(list(param.shape))

        # Reshape and assign
        param.data = torch.reshape(
            new_flat_params[start_idx:end_idx], param.shape
        )

        start_idx = end_idx



""" Optimization Algorithms """
def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
    """
    Conjugate gradient algorithm to solve Ax = b
    Used in TRPO for efficient Hessian-vector product computation

    Args:
        Av_func: Function that computes A @ v (Hessian-vector product)
        b: Target vector
        max_iter: Maximum iterations (default: 10)
        residual_tol: Residual tolerance for early stopping (default: 1e-10)

    Returns:
        x: Solution vector
    """
    # Initialize
    x = torch.zeros_like(b)
    r = b - Av_func(x)  # Residual
    p = r  # Search direction
    rsold = r.norm() ** 2

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)  # Step size

        # Update solution and residual
        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = r.norm() ** 2

        # Check convergence
        if torch.sqrt(rsnew) < residual_tol:
            break

        # Update search direction
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def rescale_and_linesearch(
    g, s, Hs, max_kl, L, kld, old_params, pi, max_iter=10,
    success_ratio=0.1
):
    """
    Backtracking line search with KL constraint (TRPO)
    Finds step size that improves objective while satisfying KL constraint

    Args:
        g: Gradient vector
        s: Search direction (from conjugate gradient)
        Hs: Hessian @ s
        max_kl: Maximum KL divergence allowed
        L: Objective function (surrogate loss)
        kld: KL divergence function
        old_params: Current parameters
        pi: Policy network
        max_iter: Maximum backtracking iterations (default: 10)
        success_ratio: Minimum improvement ratio (default: 0.1)

    Returns:
        new_params: Updated parameters (or old_params if search fails)
    """
    # Reset to old parameters and get baseline objective
    set_params(pi, old_params)
    L_old = L().detach()

    # Initial step size (based on KL constraint)
    beta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))

    # Backtracking line search
    for _ in range(max_iter):
        # Try new parameters
        new_params = old_params + beta * s

        set_params(pi, new_params)
        kld_new = kld().detach()

        L_new = L().detach()

        # Check improvement
        actual_improv = L_new - L_old
        approx_improv = torch.dot(g, beta * s)
        ratio = actual_improv / approx_improv

        # Accept if: sufficient improvement + KL constraint satisfied
        if ratio > success_ratio \
            and actual_improv > 0 \
                and kld_new < max_kl:
            return new_params

        # Reduce step size
        beta *= 0.5

    # Line search failed
    logger.warning("Line search failed! Returning old parameters")
    return old_params

# EOS - End of Script

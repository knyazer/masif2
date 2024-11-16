"""
This is a reimplementation of the IfBO:
    In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization: https://openreview.net/forum?id=VyoY3Wh9Wd
"""

## This part is the training of FT-PFN
## we start with the setup of the prior
import equinox as eqx
import jax.random as jr


def hypercube_to_params(hypercube):
    # function \pi_curve: generates combined hyperparameters given point in a hypercube
    class RandomMLP(eqx.Module):
        l1: eqx.nn.Linear
        l2: eqx.nn.Linear
        l3: eqx.nn.Linear

        def __init__(self, key):
            k1, k2, k3 = jr.split(key, 3)
            self.l1 = eqx.nn.Linear(hypercube.shape[0], 10, key=k1)
            self.l2 = eqx.nn.Linear(10, 20, key=k2)
            self.l3 = eqx.nn.Linear(20, 22, key=k3)  # 22 is the number from the paper

        def __call__(self, x):
            x = self.l1(x)

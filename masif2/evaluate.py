"""
Given a particular stored masif file, we just run the evaluation :)
since we want to "hide" part of the data, we just "enforce" a 50/50 split,
by odd/even rows. So, even rows are "test" and odd rows are "train" (the first row is train)

start by preloading all the data, such that it is stored in tuples (hypers, data, length)
where hypers is a normalized to hypercube, K-dimensional (K <= 10) array, and data
is the curve, clipped to 0-1 (with curves with nans being dropped) and subsampled/supersampled to
be 50 points long, and length is an integer specifying how much of the curve
is observed (since there are some curves in lcbench that are only 25 long (I think?), and some
curves in taskset that are 51 (maybe?) but yeah, just sort of fixing these boundary cases.
the curves are padded with ones

for evaluation, we take N random curves in the HP space, and use them as context.
"""

import os
import numpy as np
import pandas as pd
import equinox as eqx
from .main import MASIF, PiConfigSet
from .ifbo import PFN_MODEL as IFBO_PFN
from jax import random as jr
from jax import numpy as jnp
import jax
import torch

from .common import eval_model


def load_model(model_name="masif_learned_0.97.eqx", kind="learned"):
    sample_hypercube_hp = lambda k: jr.uniform(k, shape=(10,))
    pi_config = PiConfigSet(sample_hypercube_hp, jr.PRNGKey(0))
    masif = MASIF(jr.PRNGKey(1), pi_config=pi_config, kind=kind)

    model = eqx.tree_deserialise_leaves(model_name, masif)
    model = eqx.nn.inference_mode(model)
    return model


if __name__ == "__main__":
    for ctx in [400, 800, 1600, 3200, 6400]:
        for model, prefix in [
            (load_model("masif_learned_0.97.eqx"), "learned"),
            (load_model("masif_covariance.eqx"), "covariance"),
            (load_model("masif_identity.eqx"), "identity"),
            (IFBO_PFN(), "ifbo"),
        ]:
            is_ifbo = prefix == "ifbo"
            res = eval_model(model, is_ifbo, ctx, prefix)
            print(prefix, ctx, res)

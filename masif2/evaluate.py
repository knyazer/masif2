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
import functools
from tqdm import tqdm

from .common import eval_model


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"


@functools.lru_cache
def load_model(model_name, kind):
    sample_hypercube_hp = lambda k: jr.uniform(k, shape=(10,))
    pi_config = PiConfigSet(sample_hypercube_hp, jr.PRNGKey(0))
    masif = MASIF(jr.PRNGKey(1), pi_config=pi_config, kind=kind)

    model = eqx.tree_deserialise_leaves(model_name, masif)
    model = eqx.nn.inference_mode(model)
    return model


if __name__ == "__main__":
    N_ALLOC = 1000
    for ctx in tqdm([200, 400, 800, 1600]):
        for model, prefix, alloc in [
            (load_model("masif_learned_0.97.eqx", kind="learned"), "learned", N_ALLOC),
            (load_model("masif_covariance.eqx", kind="covariance"), "covariance", N_ALLOC),
            (load_model("masif_identity.eqx", kind="identity"), "identity", N_ALLOC),
            (IFBO_PFN(), "ifbo", N_ALLOC),
        ]:
            is_ifbo = prefix == "ifbo"
            res = eval_model(
                model,
                is_ifbo,
                context_points=ctx,
                name=prefix,
                shortened=False,
                num_allocations=alloc,
            )

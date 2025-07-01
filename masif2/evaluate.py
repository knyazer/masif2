import os
import numpy as np
import pandas as pd
import equinox as eqx
from .main import MASIF, PiConfigSet, load_model
from .ifbo import PFN_MODEL as IFBO_PFN
from jax import random as jr
from jax import numpy as jnp
import jax
import torch
import functools
from tqdm import tqdm

from .common import eval_model


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"


if __name__ == "__main__":
    N_ALLOC = 1000
    ctx_variants = [200, 400, 800, 1600, 3200]  # [200, 400, 800, 1600, 3200]
    """
    for ctx in ctx_variants:
        for model, prefix in [
            (load_model("models/masif_learned.eqx", kind="learned"), "learned"),
            (load_model("models/masif_covariance.eqx", kind="covariance"), "covariance"),
            (IFBO_PFN("models/ifbopfn.pt"), "ifbo"),
        ]:
            is_ifbo = prefix == "ifbo"
            res = eval_model(
                model,
                is_ifbo,
                context_points=ctx,
                name=prefix,
                shortened=False,
                benchmarks=["taskset"],
                num_allocations=N_ALLOC,
                override=True,
            )
    """

    root = "finetuned/taskset"
    for ctx in ctx_variants:
        for method in ["learned"]:
            for kind in ["comb"]:
                for n_data in [100, 800, 1600]:
                    name = f"{root}/{method}/{kind}/{n_data}"
                    model = load_model(f"models/{name}/model", kind=method)
                    res = eval_model(
                        model,
                        IFBO=False,
                        context_points=ctx,
                        benchmarks=["taskset"],
                        name=f"{name}/{ctx}",
                        shortened=False,
                        num_allocations=N_ALLOC,
                        override=True,
                    )

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

from .common import load_dataset, distance_weights


def load_model(model_name="masif_learned.eqx"):
    sample_hypercube_hp = lambda k: jr.uniform(k, shape=(10,))
    pi_config = PiConfigSet(sample_hypercube_hp, jr.PRNGKey(0))
    masif = MASIF(jr.PRNGKey(1), pi_config=pi_config)

    model = eqx.tree_deserialise_leaves(model_name, masif)
    model = eqx.nn.inference_mode(model)
    return model


def convert_to_ifbo_format(
    context_hyps, context_curves, context_lengths, target_hyp, target_len, target_value
):
    """
    Convert from IFBO format to PFN model format.

    Args:
        context_hyps: Array of shape [n_curves, n_hyps] e.g. (20, 10)
        context_curves: Array of shape [n_curves, curve_length, 1] e.g. (20, 50, 1)
        context_lengths: Array of shape [n_curves] e.g. (20,)
        target_hyp: Array of shape [n_hyps] e.g. (10,)
        target_len: Integer scalar e.g. ()
        target_value: Float scalar e.g. ()

    Returns:
        x_train: PyTorch tensor for training inputs
        y_train: PyTorch tensor for training targets
        x_test: PyTorch tensor for test inputs
        y_test: PyTorch tensor for test targets
    """

    # Convert JAX arrays to numpy if needed
    if hasattr(context_hyps, "device"):  # Check if it's a JAX array
        context_hyps = np.array(context_hyps)
        context_curves = np.array(context_curves)
        context_lengths = np.array(context_lengths)
        target_hyp = np.array(target_hyp)
        target_len = (
            np.array(target_len) if hasattr(target_len, "shape") else np.array([target_len])
        )
        target_value = (
            np.array(target_value) if hasattr(target_value, "shape") else np.array([target_value])
        )

    # Get dimensions
    n_curves = context_hyps.shape[0]
    max_length = context_curves.shape[1]  # Assuming curve_length is 50

    # Create training data for each observed point in each curve
    x_train_list = []
    y_train_list = []

    for curve_idx in range(n_curves):
        # Get the length of this curve
        curve_length = int(context_lengths[curve_idx])

        # For each observed point in the curve
        for t in range(1, curve_length + 1):  # Indices 1 to curve_length
            # Create feature vector: [curve_id, fidelity, *hyperparameters]
            # Curve ID starts at 1
            fidelity = t / max_length  # Normalize to [0, 1]

            x_row = np.concatenate([[curve_idx + 1, fidelity], context_hyps[curve_idx]])
            x_train_list.append(x_row)

            # Get the corresponding y value (performance)
            y_row = context_curves[curve_idx, t - 1, 0]  # t-1 because t starts at 1
            y_train_list.append(y_row)

    # Stack to create final training arrays
    x_train = np.vstack(x_train_list)
    y_train = np.array(y_train_list)

    # Create test data for target point
    # Determine if target hyperparameters match any context curve
    curve_id = 0  # Default to new curve

    for curve_idx in range(n_curves):
        if np.allclose(target_hyp, context_hyps[curve_idx]):
            curve_id = curve_idx + 1  # Use matched curve ID
            break

    # Target fidelity
    target_len_val = target_len.item() if hasattr(target_len, "item") else target_len
    fidelity = target_len_val / max_length

    # Create test feature vector
    x_test = np.expand_dims(np.concatenate([[curve_id, fidelity], target_hyp]), axis=0)

    # Target value
    y_test = np.array([target_value.item() if hasattr(target_value, "item") else target_value])

    # Convert to PyTorch tensors
    return (
        torch.FloatTensor(x_train),
        torch.FloatTensor(y_train),
        torch.FloatTensor(x_test),
        torch.FloatTensor(y_test),
    )


IFBO = True

if __name__ == "__main__":
    if IFBO:
        model = IFBO_PFN()
    else:
        model = load_model()

    num_allocations = 50
    master_key = jr.key(0)

    benchmarks = ["lcbench"]  # , "taskset", "pd1"]

    eval_fn = eqx.filter_jit(model.eval)

    for benchmark in benchmarks:
        lls = []

        for ds_path in os.listdir(benchmark):
            _, test = load_dataset(f"{benchmark}/{ds_path}")
            print(f"{benchmark}/{ds_path} total samples:\t", len(test))

            hyps = jnp.asarray([x[0] for x in test], dtype=jnp.float32)
            curves = jnp.asarray([x[1] for x in test], dtype=jnp.float32)
            lengths = jnp.asarray([x[2] for x in test], dtype=jnp.float32)

            # Five tensors we batch up before passing to model.eval
            accum = [[], [], [], [], []]
            lls = []

            for _ in range(num_allocations):
                master_key, k_target, k_ctx, k_len, k_eval, k_ctl, k_ctx2 = jr.split(master_key, 7)

                # ---------------- target point ----------------
                target_idx = jr.choice(k_target, len(test), shape=())  # scalar
                target_hyp = hyps[target_idx]
                target_curve = curves[target_idx]
                max_len = int(lengths[target_idx])

                # ---------------- context set -----------------
                ctx_size = int(1800 // 50)
                ctx_size = jr.randint(k_ctx, (), 1, ctx_size)

                pool_idx = jnp.arange(len(test))
                pool_hyps = hyps[pool_idx]

                prob = distance_weights(target_hyp, pool_hyps)
                prob = prob.at[target_idx].set(0)
                ctx_idx_rel = jr.choice(k_ctx2, pool_idx, shape=(ctx_size,), replace=False, p=prob)

                context_hyps = hyps[ctx_idx_rel]
                context_curves = curves[ctx_idx_rel]
                context_lengths = jr.randint(k_ctl, (ctx_size,), 1, lengths[ctx_idx_rel][0])

                # ---------------- length -----------------------
                u = jr.uniform(k_len, ())  # U(0,1)
                tgt_len = jnp.floor(jnp.exp(u * jnp.log(max_len))).astype(jnp.int32) + 1
                tgt_len = jnp.minimum(tgt_len - 1, max_len - 1)

                # --- padding ---
                input_hyp_n = 10
                pad_n = input_hyp_n - target_hyp.shape[0]

                context_hyps = jnp.concatenate(
                    [
                        context_hyps,
                        jnp.zeros((context_hyps.shape[0], pad_n), dtype=context_hyps.dtype),
                    ],
                    axis=1,
                )
                target_hyp = jnp.concatenate(
                    [target_hyp, jnp.zeros((pad_n,), dtype=target_hyp.dtype)], axis=0
                )

                if IFBO:
                    inp = convert_to_ifbo_format(
                        context_hyps,
                        context_curves[..., None],
                        context_lengths,
                        target_hyp,
                        jnp.array([tgt_len]),
                        jnp.array([target_curve[tgt_len]]),
                    )
                    mu, var = model.predict_mean_variance(*inp[:-1])
                    mu = mu.numpy().mean()
                    var = var.numpy().mean()

                    ll = float(-model.nll_loss(*inp).detach().numpy().mean())
                else:
                    inp = (
                        context_hyps,
                        context_curves[..., None],
                        context_lengths,
                        target_hyp,
                        jnp.array([tgt_len]),
                        jnp.array([target_curve[tgt_len]]),
                    )

                    ll, mu, var = eval_fn(*inp)

                lls.append(ll)
                print(
                    f"{np.array(lls).mean():.2f}({np.median(np.array(lls)):.2f}) \t {ll:.1f} \t {mu:.2f}+-{np.sqrt(var):.2f} == {inp[-1][0]:.2f}"
                )

        print(f"Mean result for {benchmark}: {jnp.stack(lls).mean():.6f}")

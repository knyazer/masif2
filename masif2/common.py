import numpy as np
import pandas as pd
from jax import random as jr
from jax import numpy as jnp
import jax


import os
import numpy as np
import pandas as pd
import equinox as eqx
from jax import random as jr
from jax import numpy as jnp
import jax
import torch


def load_dataset(name):  # noqa
    """
    Returns two lists of tuples: (hypers, data, length) for train/test splits.

    Each tuple consists of:
      - hypers: a NumPy array of normalized hyperparameters (each normalized to [0,1])
      - data: a processed curve (clipped to [0,1] and resampled/padded to exactly 50 points)
      - length: the original observed length of the curve

    The function automatically detects the dataset type from the filename:

    1. lcbench:
       - Hyperparameters:
         * batch_size: integer in [16, 512] (log scale)
         * learning_rate: continuous in [0.0001, 0.1] (log scale)
         * max_dropout: continuous in [0.0, 1.0]
         * max_units: integer in [64, 1024] (log scale)
         * momentum: continuous in [0.1, 0.99]
         * num_layers: integer in [1, 5]
         * weight_decay: continuous in [1e-05, 0.1] (log scale)

    2. taskset:
       - Hyperparameters:
         * beta1: continuous in [0.0001, 1.0] (log scale)
         * beta2: continuous in [0.001, 1.0] (log scale)
         * epsilon: continuous in [1e-12, 1000.0] (log scale)
         * learning_rate: continuous in [1e-09, 10.0] (log scale)
         * exponential_decay: continuous in [9e-07, 0.0001] (log scale)
         * l1: continuous in [1e-09, 10.0] (log scale)
         * l2: continuous in [1e-09, 10.0] (log scale)
         * linear_decay: continuous in [1e-08, 0.0001] (log scale)
         - Note: Sometimes the last four parameters may be dropped; (adam4p vs adam8p)

    3. pd1:
       - Hyperparameters:
         * lr_decay_factor: continuous in [0.01, 0.99] (linear scale) — if present
         * lr_hparams.initial_value: continuous in [1e-05, 10.0] (log scale)
         * lr_hparams.power: continuous in [0.1, 2.0] (linear scale)
         * opt_hparams.momentum: continuous in [1e-05, 1.0] (log scale)

    The curve data (stored in the "data" column) is assumed to be a NumPy array. It is processed by:
      - Converting to a NumPy array if not already one.
      - Dropping curves with any NaN values.
      - Clipping the curve values to the [0, 1] range.
      - Resampling (via linear interpolation) to exactly 50 points if longer than 50, or padding with ones if shorter.
      - Recording the original observed length.

    Finally, the dataset is split 50/50 using 1-indexed row positions (i.e. row 1, 3, 5, … are training; row 2, 4, 6, … are testing).
    """
    # Read the CSV file (assumes gzip compression)
    df = pd.read_csv(name, compression="gzip")

    # Define hyperparameters and normalization ranges for each dataset type
    # True/False corresponds to whether log normalize or not
    if "lcbench/" in name:
        hyper_cols = [
            "batch_size",
            "learning_rate",
            "max_dropout",
            "max_units",
            "momentum",
            "num_layers",
            "weight_decay",
        ]
        norm_ranges = {
            "batch_size": (16, 512, True),
            "learning_rate": (0.0001, 0.1, True),
            "max_dropout": (0.0, 1.0, False),
            "max_units": (64, 1024, True),
            "momentum": (0.1, 0.99, False),
            "num_layers": (1, 5, False),
            "weight_decay": (1e-05, 0.1, True),
        }

    elif "taskset/" in name:
        potential_cols = [
            "beta1",
            "beta2",
            "epsilon",
            "learning_rate",
            "exponential_decay",
            "l1",
            "l2",
            "linear_decay",
        ]
        hyper_cols = [col for col in potential_cols if col in df.columns]
        full_norm_ranges = {
            "beta1": (0.0001, 1.0, True),
            "beta2": (0.001, 1.0, True),
            "epsilon": (1e-12, 1000.0, True),
            "learning_rate": (1e-09, 10.0, True),
            "exponential_decay": (9e-07, 0.0001, True),
            "l1": (1e-09, 10.0, True),
            "l2": (1e-09, 10.0, True),
            "linear_decay": (1e-08, 0.0001, True),
        }
        norm_ranges = {col: full_norm_ranges[col] for col in hyper_cols}

    elif "pd1/" in name:
        hyper_cols = []
        norm_ranges = {}
        if "lr_decay_factor" in df.columns:
            hyper_cols.append("lr_decay_factor")
            norm_ranges["lr_decay_factor"] = (0.01, 0.99, False)
        if "lr_hparams.initial_value" in df.columns:
            hyper_cols.append("lr_hparams.initial_value")
            norm_ranges["lr_hparams.initial_value"] = (1e-05, 10.0, True)
        if "lr_hparams.power" in df.columns:
            hyper_cols.append("lr_hparams.power")
            norm_ranges["lr_hparams.power"] = (0.1, 3.0, False)
        if "opt_hparams.momentum" in df.columns:
            hyper_cols.append("opt_hparams.momentum")
            norm_ranges["opt_hparams.momentum"] = (1e-05, 1.0, True)
        if not hyper_cols:
            raise ValueError("No valid hyperparameters found for pd1 dataset.")

    else:
        raise ValueError("Unknown dataset type in filename.")

    train_data = []
    test_data = []
    target_length = 50  # Desired length for processed curves

    cols_to_convert = [col for col in df.columns if col != "data"]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")

    # Process each row of the dataframe
    for idx, row in df.iterrows():
        # --- Process hyperparameters ---
        hypers = []
        valid_row = True
        for col in hyper_cols:
            if col not in row or pd.isnull(row[col]):
                valid_row = False
                break
            val = row[col]
            min_val, max_val, use_log = norm_ranges[col]
            if use_log:
                if val <= 0:
                    norm_val = 0.0
                else:
                    norm_val = (np.log10(val) - np.log10(min_val)) / (
                        np.log10(max_val) - np.log10(min_val)
                    )
            else:
                try:
                    norm_val = (val - min_val) / (max_val - min_val)
                except Exception:
                    breakpoint()
            hypers.append(norm_val)
        if not valid_row:
            continue
        hypers = np.clip(np.nan_to_num(np.array(hypers)), 0, 1)

        # --- Process curve data ---
        # Here we assume the curve data is already stored as a NumPy array.
        curve = row["data"]
        if not isinstance(curve, np.ndarray):
            try:
                curve = np.asarray(curve, dtype=float)
            except Exception:
                try:
                    data_list = [float(x) for x in curve.strip()[1:-1].split()]
                    curve = np.array(data_list, dtype=float)
                except Exception:
                    try:
                        curve = "".join("" if c == "," else c for c in curve)
                        data_list = [float(x) for x in curve.strip()[1:-1].split()]
                        curve = np.array(data_list, dtype=float)
                    except Exception:
                        breakpoint()

        # Skip curves that contain any NaN values
        if np.isnan(curve).any():
            continue

        if "lcbench/" in name:
            # for some reason lcbench reports accuracy, but calls it "error_rate"
            curve = curve / 100.0
            # also, following IFBO we remove the first observation (no other reason btw)
            curve = curve[1:]
        if "taskset/" in name:
            # already rescaled (look into load_taskset.py)
            # also questionable, consult the paper for the full discussion
            pass
        if "pd1/" in name:
            # does not need anything additional done to it?
            pass
        curve = np.clip(curve, 0, 1)

        # Resample or pad the curve to have exactly target_length (50) points
        curve_len = len(curve)
        if curve_len < target_length:
            new_curve = np.concatenate([curve, np.ones(target_length - curve_len) * curve[-1]])
        else:
            new_curve = curve[:target_length]

        tup = (hypers, new_curve, min(curve_len, 50))

        # --- Split into training and testing ---
        if idx % 2 == 0:
            train_data.append(tup)
        else:
            test_data.append(tup)

    return train_data, test_data


def stick_break_length(key, max_len):
    """Sample an integer in [0, max_len-1] via DP stick breaking."""
    key_a, key_b, key_idx = jr.split(key, 3)

    alpha = jnp.exp(jr.uniform(key_a, (), minval=-4.0, maxval=-1.0))
    betas = jr.beta(key_b, a=1.0, b=alpha, shape=(max_len,))  # (L,)

    # Stick-breaking weights w_k = β_k · Π_{i<k}(1-β_i)
    residual = jax.lax.cumprod(1.0 - betas, axis=0)  # (L,)
    weights = betas * residual
    weights = weights / weights.sum()  # normalise

    return jr.choice(key_idx, jnp.arange(max_len), p=weights)  # scalar


def distance_weights(target_hyp, candidate_hyps):
    """
    Rank-based weights:
        rank 0  (closest)  → weight 1
        rank N-1 (farthest) → weight 1 / N
    Then normalise so the weights form a probability vector.
    """
    dists = jnp.linalg.norm(candidate_hyps - target_hyp, axis=-1)  # (N,)
    N = dists.shape[0]

    order = jnp.argsort(dists)  # ascending distances
    ranks = jnp.empty_like(order)  # invert permutation
    ranks = ranks.at[order].set(jnp.arange(N))  # ranks[i] = 0 … N-1

    w = 1.0 - ranks / N  # 1, 1-1/N, …, 1/N
    return w / w.sum()  # normalised probs


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


def eval_model(model, IFBO=False, context_points=900, name="default"):
    num_allocations = 50
    master_key = jr.key(0)

    store = {}
    benchmarks = ["lcbench", "taskset", "pd1"]

    eval_fn = eqx.filter_jit(model.eval)
    means = []
    meds = []

    for benchmark in benchmarks:
        store[benchmark] = []
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
                ctx_size = int(context_points // 25)
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
            means.append(np.array(lls).mean())
            meds.append(np.median(np.array(lls)))
            store[benchmark].append(np.array(lls))

        print(
            f"Mean result for {benchmark}: {np.array(means).mean():.3f}/{np.median(np.array(meds)):.3f}"
        )

    return np.array(means).mean(), np.array(meds).mean()

import numpy as np
import pandas as pd
from jax import random as jr
from jax import numpy as jnp
import jax


import os
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
from tabulate import tabulate
import random
from tqdm import tqdm

import os
import numpy as np
import pandas as pd
import equinox as eqx
from jax import random as jr
from jax import numpy as jnp
import jax
import torch
import functools
from typing import Any

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


def partition(tup, K=None):
    assert K is not None
    assert len(tup[0]) % K == 0
    out = []
    step = len(tup[0]) // K
    for i in range(0, len(tup[0]), step):
        out.append([t[i : i + step] for t in tup])
    return out


def make_single_sample(data, seed, context_points):
    hyps = np.asarray([x[0] for x in data], dtype=np.float32)
    curves = np.asarray([x[1] for x in data], dtype=np.float32)
    lengths = np.asarray([x[2] for x in data], dtype=np.float32)

    np.random.seed(seed)

    target_idx = np.random.choice(len(data))
    target_hyp, target_curve = hyps[target_idx], curves[target_idx]
    target_curve = curves[target_idx]
    max_len = int(lengths[target_idx])

    max_ctx_size = int(context_points // 25)
    ctx_size = np.random.randint(1, max_ctx_size)

    pool_idx = np.arange(len(data))

    prob = distance_weights(target_hyp, hyps[pool_idx])
    prob[target_idx] = 0
    prob /= prob.sum()

    ctx_idx_rel = np.random.choice(pool_idx, size=ctx_size, replace=True, p=prob)

    context_hyps = hyps[ctx_idx_rel]
    context_curves = curves[ctx_idx_rel]
    context_lengths = np.random.randint(1, max(lengths[ctx_idx_rel][0], 2), size=ctx_size)

    u = np.random.uniform()
    tgt_len = int(np.floor(np.exp(u * np.log(max_len)))) + 1
    tgt_len = min(tgt_len - 1, max_len - 1)

    input_hyp_n = 10
    pad_n = input_hyp_n - target_hyp.shape[0]

    context_hyps = np.concatenate(
        [
            context_hyps,
            np.zeros((context_hyps.shape[0], pad_n), dtype=context_hyps.dtype),
        ],
        axis=1,
    )

    target_hyp = np.concatenate(
        [
            target_hyp,
            np.zeros((pad_n,), dtype=target_hyp.dtype),
        ],
        axis=0,
    )

    inp = [
        pad_to_ctx(context_hyps, max_ctx_size),
        pad_to_ctx(context_curves, max_ctx_size)[..., None],
        pad_to_ctx(context_lengths, max_ctx_size),
        target_hyp,
        np.array([tgt_len]),
        np.array([target_curve[tgt_len]]),
        np.arange(max_ctx_size) < ctx_size,
    ]
    return inp


def make_seed_from_key(key):
    return int(jr.randint(key, (), 1, 1_000_000_000))


def make_batch(*, seed, size, data, context_points, wrapped=False):
    random.seed(seed)
    outs = []
    for i in range(size):
        if wrapped:
            ds = random.choice(data)
        else:
            ds = data
        out = make_single_sample(ds, seed + i, context_points)
        outs.append(out)

    stacked_outs = [[] for _ in range(len(outs[0]))]
    for x in outs:  # do a tree map
        for i, v in enumerate(x):
            stacked_outs[i].append(v)
    for i in range(len(stacked_outs)):
        stacked_outs[i] = np.array(stacked_outs[i])  # type:ignore

    return (*stacked_outs,)


def pad_to_ctx(x, N):
    padding = N - x.shape[0]
    return np.concatenate(
        [
            x,
            np.zeros((padding, *x.shape[1:]), dtype=x.dtype),
        ],
        axis=0,
    )


@functools.lru_cache
def load_dataset(name):  # noqa
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
    target_length = 50  # Desired length for processed curves

    cols_to_convert = [col for col in df.columns if col != "data"]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")

    # Process each row of the dataframe
    for _, row in df.iterrows():
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

        train_data.append(tup)

    return train_data


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
    dists = np.linalg.norm(candidate_hyps - target_hyp, axis=-1)  # (N,)
    N = dists.shape[0]

    order = np.argsort(dists)  # ascending distances
    ranks = np.empty_like(order)
    ranks[order] = np.arange(N)  # invert permutation

    w = 1.0 - ranks / N  # 1, 1-1/N, ..., 1/N
    return w / w.sum()


def convert_to_ifbo_format(
    context_hyps, context_curves, context_lengths, target_hyp, target_len, target_value, mask
):
    device = torch.device("cuda")

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
            if not mask[curve_idx]:
                continue
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
        torch.FloatTensor(x_train).to(device),
        torch.FloatTensor(y_train).to(device),
        torch.FloatTensor(x_test).to(device),
    )


def eval_model(
    model,
    IFBO=False,
    context_points=400,
    benchmarks=None,
    name="default",
    shortened=True,
    num_allocations=200,
    override=False,
):
    if benchmarks is None:
        benchmarks = ["lcbench", "pd1", "taskset"]
        if shortened:
            benchmarks = ["lcbench"]

    grand_means, grand_meds = [], []

    for ds_key_seed, benchmark in enumerate(benchmarks):
        master_key = jr.key(ds_key_seed)
        folders = os.listdir(benchmark)
        folders.sort()
        folders = folders[len(folders) // 2 :]

        for ds_path in folders:
            subbench = ds_path.replace(".", "_")

            out_dir = Path("results") / Path(name) / benchmark / Path(subbench)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"ctx_{context_points}.npz"
            b_path = Path("results") / Path(name) / "borders.npz"

            if out_path.exists() and not override:
                print("Skipping {benchmark} cuz already exists and no override flag")
                continue

            test = load_dataset(f"{benchmark}/{ds_path}")

            jax_batch = make_batch(
                seed=0, size=num_allocations, data=test, context_points=context_points
            )
            targets = jax_batch[5]  # lol, wow, so not fragile i am insane

            if IFBO:
                logits = []
                lls = []
                for val in tqdm(zip(*jax_batch)):
                    ifbo_inp = convert_to_ifbo_format(*val)
                    _logits = model.forward(*ifbo_inp)
                    _logits = torch.softmax(_logits, dim=-1).detach().cpu().numpy().squeeze()
                    logits.append(_logits)
                    lls.append(jnp.nan)
                _borders = model.model.criterion.borders.detach().cpu().numpy().squeeze()
                borders = _borders
            else:
                lls, logits = [], []
                K = max(int(context_points * num_allocations // 500_000), 1)
                if K > 50:
                    K = 200
                elif K > 10:
                    K = 50
                elif K > 5:
                    K = 10
                elif K > 2:
                    K = 4
                for batch in partition(jax_batch, K=K):
                    _lls, _logits = eqx.filter_vmap(model.eval)(*batch)
                    lls.append(_lls)
                    logits.append(_logits)
                lls = jnp.concatenate(lls)
                logits = jnp.concatenate(logits)
                borders = model.get_borders()

            print(f"Writing {out_path}")
            np.savez_compressed(
                out_path,
                np.array(logits).astype(np.float16),
                np.array(targets).astype(np.float16),
            )
            np.savez_compressed(b_path, np.array(borders).astype(np.float32))
            mean_ll, med_ll = float(np.mean(lls)), float(np.median(lls))
            grand_means.append(mean_ll)
            grand_meds.append(med_ll)

        if len(grand_means) == 0:
            print(f"Skipping the whole {benchmark}")
            continue
        print(
            f"\nMean result for {benchmark}: {np.mean(grand_means):.3f}/{np.mean(grand_meds):.3f}\n"
        )

    return float(np.mean(grand_means)), float(np.mean(grand_meds))

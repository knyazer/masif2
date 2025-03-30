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
from jax import random as jr
from jax import numpy as jnp


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


if __name__ == "__main__":
    # load a particular MASIF model
    model_name = "masif.eqx"

    sample_hypercube_hp = lambda key: jr.uniform(key, shape=(10,))
    pi_config = PiConfigSet(sample_hypercube_hp, jr.key(0))
    masif = MASIF(jr.key(1), pi_config=pi_config)
    model = eqx.tree_deserialise_leaves(model_name, masif)
    model = eqx.nn.inference_mode(model)

    batch_size = 500  # just in case of a small gpu (models are about 10M params -> should fit)
    num_allocations = 500  # too lazy to do more: the estimations are pretty accurate anyways
    key = jr.key(0)

    benchmarks = ["taskset", "pd1", "lcbench"]
    for benchmark in benchmarks:
        lls = []
        for dataset_path in os.listdir(benchmark):
            _, test = load_dataset(f"{benchmark}/{dataset_path}")
            print(f"{benchmark}/{dataset_path} total samples:\t", len(test))
            hyps = np.array([x[0] for x in test], dtype=np.float32)
            curves = np.array([x[1] for x in test], dtype=np.float32)
            lengths = np.array([x[2] for x in test], dtype=np.float32)

            accum = [[], [], [], [], []]
            for i in range(num_allocations):
                # sample random indices uniformly from test set without replacement
                context_size = 8  # exactly 400 points (first row in the ifbo table)
                indices = np.random.choice(len(test), size=context_size + 1, replace=False)
                inp_indices, target_index = indices[:-1], indices[-1]
                context_hyps = hyps[inp_indices]
                context_curves = curves[inp_indices]
                context_lengths = lengths[inp_indices]

                target_hyp = hyps[target_index]
                target_curve = curves[target_index]

                max_target_length = lengths[target_index]
                # we sample target length uniformly uniformly too!
                target_length = np.random.choice(int(max_target_length), size=1)

                accum[0].append(jnp.array(context_hyps))
                accum[1].append(jnp.array(context_curves[..., None]))
                accum[2].append(jnp.array(target_hyp))
                accum[3].append(jnp.array(target_length))
                accum[4].append(jnp.array(target_curve[target_length]))

                if len(accum[0]) >= batch_size:
                    for j in range(len(accum)):
                        accum[j] = jnp.array(accum[j])
                    key, subkey = jr.split(key)
                    log_likelihoods = eqx.filter_vmap(model.eval)(*accum)
                    lls.append(log_likelihoods.mean())
                    accum = [[], [], [], [], []]
                    print(log_likelihoods.mean())
        print(f"Mean result for {benchmark}: {np.array(lls).mean()}")

#!/usr/bin/env python
import argparse
import datetime
import io
import zipfile
from pathlib import Path

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
import requests
from einops import rearrange
from matplotlib import style
from tqdm import tqdm

from masif2 import Prior
from masif2.evals.train_lcpfn import curve_to_sample, filter_prior, make_prior, nll
from masif2.pfn import PFN, HistogramDecoder, JointEncoder

style.use("ggplot")


class Logger:
    def __init__(self, *, use_wandb=True):
        self.use_wandb = use_wandb
        if self.use_wandb:
            import wandb

            wandb.login()
            wandb.init(
                project="masif2",
                settings=wandb.Settings(code_dir="."),
                save_code=True,
                group="tpu",
            )

    def init(self, name, **kws):
        if self.use_wandb:
            import wandb

            wandb.init(
                name=name,
                project="masif2",
                settings=wandb.Settings(code_dir="."),
                reinit=True,
                group="tpu",
                **kws,
            )

    def log(self, data):
        if self.use_wandb:
            import wandb

            wandb.log(data)

    def finish(self):
        if self.use_wandb:
            import wandb

            wandb.finish()


wandb = Logger(use_wandb=False)


# Load the dataset from data/
def load_learning_curves(data_dir="data/"):
    learning_curves = []
    i = 1
    out = []
    while True:
        losses_path = Path(f"{data_dir}/losses_model_{i}.npz")
        if not losses_path.exists():
            break
        with np.load(losses_path) as data:
            losses = data["arr_0"][:, 1, ...]
            for dataset in losses:
                res = []
                for st in range(0, len(dataset), 10):
                    en = st + 10
                    interm = rearrange(
                        dataset[st:en, :, :], "batch curve data -> curve (batch data)"
                    )
                    interm = jnp.nan_to_num(interm, nan=0.0)
                    res.append(interm)
                out.append(rearrange(jnp.array(res), "a num data -> (a num) data"))
        i += 1
    learning_curves = rearrange(jnp.array(out), "b lrs data -> (b lrs) data")
    assert len(learning_curves.shape) == 2
    assert learning_curves.shape[1] == 500
    return jnp.array(learning_curves)


learning_curves_all = load_learning_curves()
holdout_size = int(0.2 * learning_curves_all.shape[0])
permutation = jr.permutation(jr.PRNGKey(0), learning_curves_all.shape[0])
learning_curves = learning_curves_all[permutation[holdout_size:]]
learning_curves_holdout = learning_curves_all[permutation[:holdout_size]]
print(
    f"Loaded {learning_curves.shape[0]} learning curves, holdout size: {holdout_size}"
)

if __name__ == "__main__":
    plt.figure(figsize=(10, 6))
    num_curves = 100
    selected_curves = jr.choice(jr.key(0), learning_curves, (num_curves,))
    for curve in selected_curves:
        plt.plot(curve, alpha=0.5)

    plt.grid(visible=True)
    plt.savefig("learning_curves_examples.png")
    plt.close()


@eqx.filter_jit
def sample_synth(prior, key, xs, n):
    curve_key, sample_key = jr.split(key, 2)
    curves = prior.sample(key=curve_key, xs=xs, n=n)
    return eqx.filter_vmap(eqx.Partial(curve_to_sample, xs=xs))(
        curves, jr.split(sample_key, n)
    )


@eqx.filter_jit
def sample_real(key, n, xs):
    samples = jr.choice(key, learning_curves[:, : xs.shape[0]], (n,))
    return eqx.filter_vmap(eqx.Partial(curve_to_sample, xs=xs))(
        samples, jr.split(key, n)
    )


@eqx.filter_jit
def sample_real_augged(key, n, xs, strength=0.1):
    # strength from 0 to 1, augments the learning curve;
    # I can make advanced augmentations later, with the distribution of sample mse
    # errors, but for now this is enough to just shift-scale the curve
    samples = jr.choice(key, learning_curves[:, : xs.shape[0]], (n,))

    shift_key, scale_key = jr.split(key)
    shifts = jr.uniform(shift_key, (n,), minval=-strength, maxval=strength)
    scales = jr.uniform(scale_key, (n,), minval=1 - strength, maxval=1 + strength)

    augmented_samples = samples * scales[:, None] + shifts[:, None]
    augmented_samples = jnp.clip(augmented_samples, 0.0, 1.0)

    return eqx.filter_vmap(eqx.Partial(curve_to_sample, xs=xs))(
        augmented_samples, jr.split(key, n)
    )


# Download and process LCBench data


def download_lcbench_data(url="https://ndownloader.figshare.com/files/21188607"):
    if (
        not Path("lcbench_data").exists()
        or not Path("lcbench_data/data_2k.json").exists()
    ):
        print("Downloading LCBench data...")
        response = requests.get(url, stream=True, timeout=60)
        total_size = int(response.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="iB", unit_scale=True) as pbar:
            content = io.BytesIO()
            for data in response.iter_content(chunk_size=1024):
                size = content.write(data)
                pbar.update(size)
        with zipfile.ZipFile(content) as zip_ref:
            zip_ref.extractall("lcbench_data")
        print("Download complete.")
    else:
        print("LCBench data already downloaded. Skipping download.")


def process_lcbench_data(padding_len=50):
    if Path("lcbench_processed.npz").exists():
        print("Loading processed LCBench data...")
        with np.load("lcbench_processed.npz") as data:
            return data["arr_0"][..., :padding_len]
    else:
        print("meh, we need to download stuff")
        download_lcbench_data()
        with Path("lcbench_data/data_2k.json").open("rb") as f:
            print("Processing LCBench data...")
            import ijson

            res = []
            c = 0
            for c, (key, value) in tqdm(enumerate(ijson.kvitems(f, ""))):
                if c >= 30:  # json 31 is invalid
                    break
                arrs = []
                for i in range(2000):
                    raw = [float(x) for x in value[str(i)]["log"]["Train/val_accuracy"]]
                    raw_array = jnp.array(raw)
                    if jnp.any(jnp.isnan(raw_array)):
                        last_non_nan = raw_array[
                            jnp.where(~jnp.isnan(raw_array))[0][-1]
                        ]
                        raw_array = jnp.nan_to_num(raw_array, nan=last_non_nan)
                    arrs.append(
                        jnp.pad(
                            raw_array,
                            (0, padding_len - len(raw)),
                            mode="constant",
                            constant_values=raw_array[-1],
                        )
                    )
                # concat arrs
                res.append(jnp.stack(arrs))
                del key, value
            res_jax = rearrange(jnp.stack(res), "a b c -> (a b) c") / 100.0
            jnp.savez("lcbench_processed.npz", res_jax)
            return res_jax


NUM_EPOCHS = 10000
BATCH_SIZE = 50
CURVE_LENGTH = 50  # LCBench curves are 50 long

SMALL_PFN = {
    "pos_embed": 12,
    "val_embed": 12,
    "num_heads": 2,
    "n_layers": 3,
    "hidden_size": 24,
    "emb_size": 24,
}

MEDIUM_PFN = {
    "pos_embed": 24,
    "val_embed": 24,
    "num_heads": 4,
    "n_layers": 6,
    "hidden_size": 48,
    "emb_size": 48,
}

LARGE_PFN = {
    "pos_embed": 48,
    "val_embed": 48,
    "num_heads": 8,
    "n_layers": 12,
    "hidden_size": 96,
    "emb_size": 96,
}

VERSION = 12


def main(
    run_name="default",
    augmentation_strength=0.1,
    lr=1e-3,
    data_split_ratio=0.5,
    pfn_config=SMALL_PFN,
):
    k1, k2, k3, k4, k5, k6 = jr.split(jr.PRNGKey(42), 6)
    xs = jnp.arange(CURVE_LENGTH).astype(jnp.float32) + 1  # start counting from 1
    prior = Prior(prior_fn=make_prior, reject_fn=filter_prior)
    test_samples = sample_synth(prior, key=k1, xs=xs, n=5_000)
    test_real_samples = eqx.filter_vmap(eqx.Partial(curve_to_sample, xs=xs))(
        learning_curves_holdout[:, : xs.shape[0]],
        jr.split(k6, learning_curves_holdout.shape[0]),
    )
    lcbench_curves = process_lcbench_data()
    lcbench_subsampled = lcbench_curves[
        jr.randint(k5, (5000,), 0, lcbench_curves.shape[0])
    ]
    lcbench_samples = eqx.filter_vmap(eqx.Partial(curve_to_sample, xs=xs))(
        lcbench_subsampled, jr.split(k6, lcbench_subsampled.shape[0])
    )

    decoder = HistogramDecoder(n_bins=100)
    decoder = decoder.fit(prior.sample(key=k2, xs=xs, n=20_000).ravel())

    assert not jnp.any(jnp.isnan(decoder.bounds))

    model = PFN(
        # encoder
        encoder=JointEncoder(
            positional_embedding_size=pfn_config["pos_embed"],
            value_embedding_size=pfn_config["val_embed"],
            key=k3,
        ),
        # 'body' of the net
        n_layers=pfn_config["n_layers"],
        hidden_size=pfn_config["hidden_size"],
        embed_size=pfn_config["emb_size"],
        num_heads=pfn_config["num_heads"],
        key=k4,
        # decoder
        decoder=decoder,
    )

    wandb.init(
        run_name,
        config={
            "augmentation_strength": augmentation_strength,
            "lr": lr,
            "data_split_ratio": data_split_ratio,
            "optimizer": "adam-rop",
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "version": VERSION,
        },
    )
    optim = optax.chain(
        optax.adam(lr),
        optax.contrib.reduce_on_plateau(
            patience=5,
            cooldown=200,
            factor=0.7,
            rtol=1e-2,
            accumulation_size=20,
        ),
    )
    opt_state = optim.init(model.params())
    loss = None
    synth_epochs_ratio = data_split_ratio

    def train_step(carry, i):
        model, opt_state = carry

        train_samples = jax.lax.cond(
            i > synth_epochs_ratio * NUM_EPOCHS,
            lambda: sample_real_augged(
                jr.PRNGKey(i), n=BATCH_SIZE, xs=xs, strength=augmentation_strength
            ),
            lambda: sample_synth(prior, key=jr.PRNGKey(i), xs=xs, n=BATCH_SIZE),
        )

        loss, grads = eqx.filter_value_and_grad(eqx.Partial(nll, sample=train_samples))(
            model
        )

        updates, opt_state = optim.update(grads, opt_state, model, value=loss)
        model = eqx.apply_updates(model, updates)

        return (model, opt_state), loss

    @eqx.filter_jit
    def evaluate(model):
        loss = nll(model, sample=test_samples)
        test_real = nll(model, sample=test_real_samples)
        test_lcbench = nll(model, sample=lcbench_samples)
        return (loss, test_real, test_lcbench)

    logging_freq = 100
    for it in (pbar := tqdm(range(NUM_EPOCHS // logging_freq))):
        (model, opt_state), training_losses = eqxi.scan(
            train_step,
            (model, opt_state),
            jnp.arange(logging_freq) + logging_freq * it,
            kind="lax",
        )

        for i, tl in enumerate(training_losses):
            wandb.log({"train_loss": tl, "epoch": it * logging_freq + i})

        loss, test_real, test_lcbench = evaluate(model)
        wandb.log(
            {
                "synth_test_loss": loss,
                "real_test_loss": test_real,
                "lcbench_loss": test_lcbench,
                "epoch": it,
                "synth": int(it > synth_epochs_ratio * NUM_EPOCHS),
                "learning_rate": optax.tree_utils.tree_get(opt_state, "scale") * lr,
            }
        )
        pbar.set_description(f"test: {test_lcbench} | train: {training_losses[-1]}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--n_hosts", type=int, default=1, help="Number of hosts for parallel execution"
    )
    parser.add_argument("--id", type=int, default=0, help="ID of the current host")
    args = parser.parse_args()

    pfn_size = args.size

    pfn_config = {"small": SMALL_PFN, "medium": MEDIUM_PFN, "large": LARGE_PFN}[
        pfn_size
    ]

    wandb = Logger(use_wandb=args.wandb)
    date = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=1))
    ).strftime("%m%d")

    lrs = [3e-3]
    strengths = [0.0, 0.1, 0.3, 0.5]
    splits = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    total_combinations = len(lrs) * len(strengths) * len(splits)
    combinations_per_host = 1 + (total_combinations + args.n_hosts - 1) // args.n_hosts
    start_idx = args.id * combinations_per_host
    end_idx = min((args.id + 1) * combinations_per_host, total_combinations)

    current_idx = 0
    for lr in lrs:
        for strength in strengths:
            for split in splits:
                if start_idx <= current_idx < end_idx:
                    main(
                        run_name=f"{date}|lr{lr}_aug{strength}_{split}-v{VERSION}|{pfn_size[0]}",
                        augmentation_strength=strength,
                        lr=lr,
                        data_split_ratio=split,
                        pfn_config=pfn_config,
                    )
                current_idx += 1

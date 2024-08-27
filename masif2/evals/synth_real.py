from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from einops import rearrange
from matplotlib import style
from tqdm import tqdm

from masif2 import Prior
from masif2.evals.train_lcpfn import curve_to_sample, filter_prior, make_prior, nll
from masif2.pfn import PFN, HistogramDecoder, JointEncoder

style.use("ggplot")


class Logger:
    def __init__(self, *, use_wandb=True, project_name="masif2"):
        self.use_wandb = use_wandb
        if self.use_wandb:
            import wandb

            wandb.login()
            wandb.init(project=project_name, settings=wandb.Settings(code_dir="."))

    def log(self, data):
        if self.use_wandb:
            import wandb

            wandb.log(data)

    def finish(self):
        if self.use_wandb:
            import wandb

            wandb.finish()


wandb = Logger(use_wandb=True)


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
    # I can make advanced augmentations later, with analyzing the distribution of sample-wise
    # errors, but for now this is enough (just shift-scale the curve, while making sure it fits in [0, 1])
    samples = jr.choice(key, learning_curves[:, : xs.shape[0]], (n,))

    shift_key, scale_key = jr.split(key)
    shifts = jr.uniform(shift_key, (n,), minval=-strength, maxval=strength)
    scales = jr.uniform(scale_key, (n,), minval=1 - strength, maxval=1 + strength)

    augmented_samples = samples * scales[:, None] + shifts[:, None]
    augmented_samples = jnp.clip(augmented_samples, 0.0, 1.0)

    return eqx.filter_vmap(eqx.Partial(curve_to_sample, xs=xs))(
        augmented_samples, jr.split(key, n)
    )


NUM_EPOCHS = 2000
BATCH_SIZE = 500

if __name__ == "__main__":
    k1, k2, k3, k4, k5, k6 = jr.split(jr.PRNGKey(42), 6)
    xs = jnp.arange(100).astype(jnp.float32) + 1
    prior = Prior(prior_fn=make_prior, reject_fn=filter_prior)
    test_samples = sample_synth(prior, key=k1, xs=xs, n=5_000)
    test_real_samples = eqx.filter_vmap(eqx.Partial(curve_to_sample, xs=xs))(
        learning_curves_holdout[:, : xs.shape[0]],
        jr.split(k6, learning_curves_holdout.shape[0]),
    )

    decoder = HistogramDecoder(n_bins=100)
    decoder = decoder.fit(prior.sample(key=k2, xs=xs, n=20_000).ravel())

    assert not jnp.any(jnp.isnan(decoder.bounds))

    model = PFN(
        # encoder
        encoder=JointEncoder(
            positional_embedding_size=12,
            value_embedding_size=12,
            key=k3,
        ),
        # 'body' of the net
        n_layers=3,
        hidden_size=24,
        embed_size=24,
        num_heads=2,
        key=k4,
        # decoder
        decoder=decoder,
    )

    optim = optax.adam(5e-4)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    loss = None
    synth_epochs_ratio = 0.5
    for i in (pbar := tqdm(range(NUM_EPOCHS))):
        if i > synth_epochs_ratio * NUM_EPOCHS:
            train_samples = sample_real_augged(jr.PRNGKey(i), n=BATCH_SIZE, xs=xs)
        else:
            train_samples = sample_synth(prior, key=jr.PRNGKey(i), xs=xs, n=BATCH_SIZE)

        _tloss, grads = eqx.filter_value_and_grad(
            eqx.Partial(nll, sample=train_samples)
        )(model)
        del train_samples

        assert jnp.allclose(grads.decoder.bounds, 0.0)
        assert jnp.allclose(grads.decoder.left_std, 0.0)
        assert jnp.allclose(grads.decoder.right_std, 0.0)

        if i % 10 == 0:
            loss = nll(model, sample=test_samples)
            test_real = nll(model, sample=test_real_samples)
            wandb.log(
                {
                    "synth_test_loss": loss,
                    "real_test_loss": test_real,
                    "train_loss": _tloss,
                    "epoch": i,
                    "synth": int(i > synth_epochs_ratio * NUM_EPOCHS),
                }
            )
        synth_symbol = "S" if i <= synth_epochs_ratio * NUM_EPOCHS else "R"
        pbar.set_description(f"{synth_symbol} | test: {loss} | train: {_tloss}")

        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

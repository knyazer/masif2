#!/usr/bin/env poetry -v run python
import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from tqdm import tqdm

from masif2 import Prior
from masif2.pfn import PFN, HistogramDecoder, JointEncoder

jax.config.update("jax_compilation_cache_dir", ".jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def pow3(x, c, a, alpha):
    return c - a * jnp.power(x, -alpha)


def janoschek(x, alpha, beta, kappa, delta):
    return alpha - (alpha - beta) * jnp.exp(-kappa * jnp.power(x, delta))


def ilog2(x, a, c):
    return c - a / jnp.log(x + 1)


def combined_prior_fn(
    x: Float[Array, "seq_len"],
    weights: Float[Array, "3"],
    theta: Float[Array, "9"],
):
    return (
        weights[0] * pow3(x, *theta[0:3])
        + weights[1] * janoschek(x, *theta[3:7])
        + weights[2] * ilog2(x, *theta[7:9])
    )


def make_prior(*, key: PRNGKeyArray, xs: Float[Array, "seq_len"]):
    # pre-condition: x starts from 1!
    xs = eqx.error_if(
        xs,
        jnp.any(xs < 1),
        "There must be only values >= 1 in the input sequence for prior generation!",
    )

    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12 = jr.split(key, 12)
    weights = jr.uniform(k1, (3,), minval=0, maxval=1)

    theta = jnp.array(
        [
            # pow_3 params
            jr.uniform(k2, (), minval=0, maxval=1.25),
            jr.uniform(k3, (), minval=-0.6, maxval=0.6),
            jnp.exp(jr.normal(k4, ()) * 2),
            # janoschek params
            jr.uniform(k5, (), minval=0, maxval=1),
            jr.uniform(k6, (), minval=0, maxval=2),
            # . jnp.exp(jr.normal(k7, ()) - 2), # copied from the paper, poor perf
            # . jnp.exp(jr.normal(k8, ()) * 0.5),
            jr.uniform(k7, (), minval=0, maxval=1),  # copied from the source code
            jr.uniform(k8, (), minval=-5, maxval=5),
            # ilog2 params
            jr.uniform(k9, (), minval=0, maxval=1),
            jr.uniform(k10, (), minval=-0.5, maxval=0.5),
        ],
    )

    std_noise = jnp.sqrt(jnp.exp(jr.normal(k11, ()) * jnp.sqrt(2) - 8))
    noise = jr.normal(k12, xs.shape) * std_noise

    return combined_prior_fn(xs, weights, theta) + noise


@eqx.filter_jit  # mandatory, since handling of bin ops is different
def filter_prior(*, sample, **_):
    not_nan = jnp.logical_not(jnp.any(jnp.isnan(sample)))
    improving = sample[0] < sample[-1]
    in_bounds = jnp.all(jnp.logical_and(sample >= 0.0, sample <= 1.0))

    # TODO: make filter return as in 'do we keep it?'
    return jnp.logical_not(
        jnp.logical_and(not_nan, jnp.logical_and(improving, in_bounds)),
    )


class Sample(eqx.Module):
    xs: Float[Array, "n"]
    ys: Float[Array, "n"]
    target_x: Float[Array, ""]
    _target_y: Float[Array, ""]
    mask: Bool[Array, "n"]


def curve_to_sample(
    curve: Float[Array, "dim"], key: PRNGKeyArray, xs: Float[Array, "dim"]
):
    n = len(curve)
    key_points, key_target = jr.split(key, 2)

    # we will generate masks as two masked intervals
    p1, p2, p3, p4 = jr.uniform(key_points, (4,), minval=0, maxval=n - 1)
    p1, p2 = jax.lax.cond(p1 < p2, lambda: (p1, p2), lambda: (p2, p1))
    p3, p4 = jax.lax.cond(p3 < p4, lambda: (p3, p4), lambda: (p4, p3))

    ind = jnp.arange(n)
    masked = ((p1 <= ind) & (ind < p2)) | ((p3 <= ind) & (ind < p4))
    masked_curve = jnp.where(
        masked,
        jnp.nan * curve,
        curve,
    )
    masked_xs = jnp.where(
        masked,
        jnp.nan * xs,
        xs,
    )

    target_token_ind = jr.categorical(
        key=key_target,
        logits=masked.astype(jnp.float32),
    )
    return Sample(
        xs=masked_xs,
        ys=masked_curve,
        target_x=xs[target_token_ind],
        _target_y=curve[target_token_ind],
        mask=jnp.bitwise_not(masked),
    )


@eqx.filter_jit
def sample(prior, key, xs, n):
    curve_key, sample_key = jr.split(key, 2)
    curves = prior.sample(key=curve_key, xs=xs, n=n)
    return eqx.filter_vmap(eqx.Partial(curve_to_sample, xs=xs))(
        curves, jr.split(sample_key, n)
    )


@eqx.filter_jit
def nll(model, sample):
    distrs = eqx.filter_vmap(model)(
        sample.xs,
        sample.ys,
        sample.mask,
        sample.target_x,
    )
    return -jnp.log(
        eqx.filter_vmap(lambda distr, target_y: distr.pdf(target_y))(
            distrs,
            sample._target_y,  # noqa: SLF001
        ),
    ).mean()


NUM_EPOCHS = 10000
BATCH_SIZE = 1000

if __name__ == "__main__":
    k1, k2, k3, k4, k5 = jr.split(jr.PRNGKey(42), 5)
    xs = jnp.arange(100).astype(jnp.float32) + 1
    prior = Prior(prior_fn=make_prior, reject_fn=filter_prior)
    test_samples = sample(prior, key=k1, xs=xs, n=5_000)

    decoder = HistogramDecoder(n_bins=500)
    decoder = decoder.fit(prior.sample(key=k2, xs=xs, n=200_000).ravel())

    assert not jnp.any(jnp.isnan(decoder.bounds))

    model = PFN(
        # encoder
        encoder=JointEncoder(
            positional_embedding_size=16,
            value_embedding_size=16,
            key=k3,
        ),
        # 'body' of the net
        n_layers=3,
        hidden_size=32,
        embed_size=32,
        num_heads=2,
        key=k4,
        # decoder
        decoder=decoder,
    )

    optim = optax.adam(5e-4)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    loss = None
    for i in (pbar := tqdm(range(NUM_EPOCHS))):
        train_samples = sample(prior, key=jr.PRNGKey(i), xs=xs, n=1_000)

        _tloss, grads = eqx.filter_value_and_grad(
            eqx.Partial(nll, sample=train_samples),
        )(
            model,
        )

        assert jnp.allclose(grads.decoder.bounds, 0.0)
        assert jnp.allclose(grads.decoder.left_std, 0.0)
        assert jnp.allclose(grads.decoder.right_std, 0.0)

        if i % 10 == 0:
            loss = nll(model, sample=test_samples)
        pbar.set_description(f"test loss : {loss} | train loss: {_tloss}")

        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

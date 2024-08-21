import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from masif2 import Prior
from masif2.pfn import PFN, HistogramDecoder, JointEncoder


def test_multistage_basic():
    def prior_fn(*, key: PRNGKeyArray, xs: Float[Array, "dim"], **_):
        k1, k2 = jr.split(key, 2)
        coeff = jr.uniform(k1, minval=-0.5, maxval=0.5)
        ys = xs * coeff + 4.2
        ys = ys + jr.normal(k2, ys.shape)
        return ys

    g_xs = jnp.arange(100).astype(jnp.float32)
    prior = Prior(prior_fn=prior_fn)
    samples = prior.sample(
        key=jr.PRNGKey(0),
        xs=g_xs,
        n=100_000,
    )

    key_test, key_net, key_encoder = jr.split(jr.PRNGKey(1), 3)

    class Sample(eqx.Module):
        xs: Float[Array, "n"]
        ys: Float[Array, "n"]
        target_x: Float[Array, ""]
        _target_y: Float[Array, ""]
        mask: Bool[Array, "n"]

    @eqx.filter_vmap
    def make_targets(single_curve: Float[Array, "dim"], key: PRNGKeyArray):
        n = len(single_curve)
        key_points, key_target = jr.split(key, 2)

        # we will generate masks as two masked intervals
        p1, p2, p3, p4 = jr.uniform(key_points, (4,), minval=0, maxval=n - 1)
        p1, p2 = jax.lax.cond(p1 < p2, lambda: (p1, p2), lambda: (p2, p1))
        p3, p4 = jax.lax.cond(p3 < p4, lambda: (p3, p4), lambda: (p4, p3))

        ind = jnp.arange(n)
        masked = ((p1 <= ind) & (ind < p2)) | ((p3 <= ind) & (ind < p4))
        masked_curve = jnp.where(
            masked,
            jnp.nan * jnp.zeros_like(single_curve),
            single_curve,
        )
        masked_xs = jnp.where(
            masked,
            jnp.nan * jnp.zeros_like(g_xs),
            g_xs,
        )

        target_token_ind = jr.categorical(
            key=key_target,
            logits=masked.astype(jnp.float32),
        )
        return Sample(
            xs=masked_xs,
            ys=masked_curve,
            target_x=g_xs[target_token_ind],
            _target_y=single_curve[target_token_ind],
            mask=jnp.bitwise_not(masked),
        )

    decoder = HistogramDecoder(n_bins=500)
    decoder = decoder.fit(samples.ravel())

    assert not jnp.any(jnp.isnan(decoder.bounds))

    pfn = PFN(
        # encoder
        encoder=JointEncoder(
            positional_embedding_size=16,
            value_embedding_size=16,
            key=key_encoder,
        ),
        # 'body' of the net
        n_layers=3,
        hidden_size=32,
        embed_size=32,
        num_heads=2,
        key=key_net,
        # decoder
        decoder=decoder,
    )

    pfn = eqx.tree_deserialise_leaves("pfn.eqx", pfn)

    beheaded = pfn.behead()
    model = MASIF2(
        encoder=beheaded,
        decoder=decoder,
        hyp_weighting=lambda target, x: 1.0 / (jnp.sum((target - x) ** 2) + 1e-4),
    )

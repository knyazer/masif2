from typing import Optional

import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr
from jax.scipy import stats
from jaxtyping import Array, Float, Int, PRNGKeyArray

from masif2.utils import MASIFError


class TransformerLayer(eqx.Module):
    """A single transformer layer."""

    attention: eqx.nn.MultiheadAttention

    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm

    def __init__(
        self,
        hidden_size: int,
        embed_size: int,
        num_heads: int,
        key: PRNGKeyArray,
        dropout_p: float = 0.1,
    ):
        attention_key, mlp_key, out_key = jax.random.split(key, 3)

        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=attention_key,
            dropout_p=dropout_p,
        )

        self.mlp = eqx.nn.Linear(hidden_size, embed_size, key=mlp_key)
        self.output = eqx.nn.Linear(embed_size, hidden_size, key=out_key)

        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)

    def __call__(
        self,
        inputs: Float[Array, "seq_len hidden_size"],
        mask: Optional[Int[Array, " seq_len"]] = None,
        attn_key: Optional[PRNGKeyArray] = jr.PRNGKey(0),
        inference: bool = False,
    ) -> Float[Array, "seq_len hidden_size"]:
        x = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            mask=mask,
            key=attn_key,
            inference=inference,
        )

        x = x + inputs  # residual connection
        x = jax.vmap(self.layernorm)(x)  # normalize

        def ff(inp):
            hidden = jax.nn.gelu(self.mlp(inp), approximate=True)  # project to embed
            output = self.output(hidden)  # project back to the original size
            output = self.layernorm(output + inp)  # add residual and normalize
            return output

        x = jax.vmap(ff)(x)  # use feedforward block on every 'token'
        return x


class HistogramDecoder(eqx.Module):
    bounds: Float[Array, "hist_size"]  # stores n_bins + 1 bounds(e.g. +inf, -inf, ...)
    left_std: Float[Array, ""]
    right_std: Float[Array, ""]
    n_bins: int

    def __init__(
        self,
        n_bins: int,
        left_std=None,
        right_std=None,
        bounds=None,
    ):
        if n_bins == 1:
            raise MASIFError("Single bin is an edge case that we explicitly ignore")
        if n_bins <= 0:
            raise MASIFError("WDYM you want {n_bins} bins? does not make sense")
        self.n_bins = n_bins
        self.left_std = jnp.array(0.0) if left_std is None else left_std
        self.right_std = jnp.array(0.0) if right_std is None else right_std
        if bounds is None:
            self.bounds = jnp.zeros((n_bins + 1,))
        else:
            self.bounds = bounds

    @eqx.filter_jit
    def fit(self, prior_samples: Float[Array, "n_samples"]):
        sorted_samples = jnp.sort(prior_samples)
        n_samples = prior_samples.shape[0]

        # make bounds according to quantiles
        bounds_indices = (
            (n_samples / self.n_bins) * (jnp.arange(1, self.n_bins))
        ).astype(jnp.int32)
        bounds_without_infs = sorted_samples[bounds_indices]
        bounds_with_infs = jnp.concatenate(
            [jnp.array([-jnp.inf]), bounds_without_infs, jnp.array([jnp.inf])],
        )

        # quick adequacy check
        assert (
            len(bounds_with_infs) == self.n_bins + 1
        ), f"{len(self.bounds)} != {self.n_bins}"

        # and, of course, compute stds
        scale_factor = 1.0 / jnp.sqrt(1 - 2 / jnp.pi)  # hard math wow
        left_std = (
            jnp.std(sorted_samples, where=sorted_samples < bounds_without_infs[0])
            * scale_factor
        )
        right_std = (
            jnp.std(sorted_samples, where=sorted_samples >= bounds_without_infs[-1])
            * scale_factor
        )
        return HistogramDecoder(self.n_bins, left_std, right_std, bounds_with_infs)

    def __call__(self, weights: Float[Array, "n_bins"]):
        assert len(weights) == len(self.bounds) - 1
        assert self.n_bins == len(weights)
        return Histogram(
            self.bounds, self.left_std, self.right_std, weights / weights.sum(),
        )


class Histogram(eqx.Module):
    bounds: Float[Array, "dim_plus_one"]
    weights: Float[Array, "dim"]
    left_std: Float[Array, ""]
    right_std: Float[Array, ""]
    n_bins: int

    def __init__(self, bounds, left_std, right_std, weights):
        self.bounds = bounds
        self.left_std = left_std
        self.right_std = right_std
        self.weights = weights
        self.n_bins = weights.shape[0]

    @eqx.filter_jit
    def pdf(self, x: Float[Array, ""]):
        def standard_bins():
            index = jnp.argmin(self.bounds <= x)
            density = self.weights[index - 1] / (
                self.bounds[index] - self.bounds[index - 1]
            )
            return density

        def half_normal_bins():
            def left_normal():
                delta = self.bounds[1] - x
                normal_pdf = stats.norm.pdf(delta, scale=self.left_std)
                return normal_pdf * self.weights[0] * 2

            def right_normal():
                delta = x - self.bounds[-2]
                normal_pdf = stats.norm.pdf(delta, scale=self.right_std)
                return normal_pdf * self.weights[-1] * 2

            return jax.lax.cond(x < self.bounds[1], left_normal, right_normal)

        likelihood = jax.lax.cond(
            jnp.logical_and(x >= self.bounds[1], x < self.bounds[-2]),
            standard_bins,
            half_normal_bins,
        )
        return likelihood

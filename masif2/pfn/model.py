from typing import Any, Callable

import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .decoders import Decoder

# interfaces
from .encoders import Encoder


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
        mask: Bool[Array, "seq_len seq_len"] | None = None,
        attn_key: PRNGKeyArray | None = None,
        /,
        inference: bool = False,  # noqa
    ) -> Float[Array, "seq_len hidden_size"]:
        if attn_key is None:
            attn_key = jr.PRNGKey(42)  # dealing with dropout nans
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


class Conditioner(eqx.Module):
    embedding: eqx.nn.Embedding
    ff: eqx.nn.Linear
    time_max: Any

    def __init__(self, time_dim, hidden_dim, *, key: PRNGKeyArray):
        k1, k2 = jr.split(key)
        self.embedding = eqx.nn.Embedding(time_dim, hidden_dim, key=k1)
        self.time_max = time_dim
        self.ff = eqx.nn.Linear(hidden_dim * 2, hidden_dim, key=k2)

    def __call__(self, time, x):
        assert time.dtype == jnp.int32
        assert len(time.shape) == 0
        time = eqx.error_if(
            time,
            time >= self.time_max,
            "time passed to conditioning is larger than limit of embedding",
        )
        time = eqx.error_if(time, time < 0, "negative time in conditioner")

        time_processed = self.embedding(time)
        out = self.ff(jnp.concatenate([time_processed, x]).ravel())
        return jax.nn.gelu(out)


class PFN(eqx.Module):
    layers: list[TransformerLayer]
    encoder: Encoder
    decoder_glue: Callable
    decoder: Callable
    conditioning: Any

    def params(self):
        s = eqx.filter(self, eqx.is_array)
        for i in range(len(self.layers)):
            s = eqx.tree_at(
                lambda x: x.layers[i].attention.dropout.p,
                s,
                self.layers[i].attention.dropout.p,
                is_leaf=lambda x: x is None,
            )
        return s

    def __init__(
        self,
        *,
        encoder: Encoder | None = None,
        n_layers: int | None = None,
        decoder: Decoder | None = None,
        key: PRNGKeyArray | None = None,
        **kws,
    ):
        # force to pass the encoder, n_layers and decoder
        assert encoder is not None
        assert n_layers is not None
        assert key is not None
        # each transformer block wants a different key
        key_layers, key_glue, key_cond = jr.split(key, 3)
        keys = jr.split(key_layers, n_layers)
        del key  # avoid shadowing
        self.encoder = encoder
        self.layers = [TransformerLayer(key=_key, **kws) for _key in keys]
        self.conditioning = Conditioner(50, kws["embed_size"], key=key_cond)
        if decoder is not None:
            self.decoder_glue = eqx.nn.Linear(
                kws["embed_size"],
                decoder.n_bins,  # type: ignore
                key=key_glue,
            )
            self.decoder = decoder
        else:
            self.decoder_glue = lambda x: x
            self.decoder = lambda x: x

    def behead(self):
        # removes the glue and the decoder
        s = self
        s = eqx.tree_at(lambda x: x.decoder_glue, s, lambda inp: inp)
        s = eqx.tree_at(lambda x: x.decoder, s, lambda inp: inp)
        return s

    def __call__(self, xs, ys, target_x):
        x = self.encoder(xs, ys)

        for layer in self.layers:
            x = layer(x, jnp.ones((x.shape[0], x.shape[0])).astype(jnp.bool))

        x = eqx.error_if(
            x,
            jnp.any(jnp.isnan(x)),
            "Nans encountered after the transformer layers",
        )
        x = eqx.filter_vmap(lambda _x: self.conditioning(target_x, _x))(x)

        x = eqx.filter_vmap(self.decoder_glue)(x)
        x = eqx.filter_vmap(self.decoder)(x)
        return x

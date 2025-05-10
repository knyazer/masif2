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
    """A single transformer layer with *causal* self‑attention."""

    attention: eqx.nn.MultiheadAttention
    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm1: eqx.nn.LayerNorm
    layernorm2: eqx.nn.LayerNorm

    def __init__(
        self,
        hidden_size: int,
        embed_size: int,
        num_heads: int,
        key: PRNGKeyArray,
        dropout_p: float = 0.0,
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

        self.layernorm1 = eqx.nn.LayerNorm(shape=hidden_size)
        self.layernorm2 = eqx.nn.LayerNorm(shape=hidden_size)

    def _build_causal_mask(self, seq_len: int) -> Bool[Array, "seq_len seq_len"]:
        """Lower‑triangular mask allowing each position to attend to itself and the past."""
        return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

    def __call__(
        self,
        inputs: Float[Array, "seq_len hidden_size"],
        mask: Bool[Array, "seq_len seq_len"] | None = None,
        attn_key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        # If no mask was supplied, fall back to a causal mask.
        if mask is None:
            mask = self._build_causal_mask(inputs.shape[0])

        if attn_key is None:
            # Separate key ensures deterministic behaviour when dropout_p == 0.0
            attn_key = jr.PRNGKey(42)

        x = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            mask=mask,
            key=attn_key,
        )

        x = x + inputs  # residual connection
        x = jax.vmap(self.layernorm1)(x)  # normalize

        def ff(inp):
            hidden = jax.nn.gelu(self.mlp(inp), approximate=True)  # project to embed
            output = self.output(hidden)  # project back to the original size
            output = self.layernorm2(output + inp)  # add residual and normalize
            return output

        x = jax.vmap(ff)(x)  # use feedforward block on every token
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

    def __call__(self, x, time):
        time = eqx.error_if(time, time - jnp.round(time) > 1e-6, "yikes yikes")
        time = time.astype(jnp.int32)
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
    conditioner: Conditioner

    # ---------------------------------------------------------------------
    #                           UTILITY METHODS
    # ---------------------------------------------------------------------

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

    # ---------------------------------------------------------------------
    #                              INIT
    # ---------------------------------------------------------------------

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
        self.conditioner = Conditioner(50, kws["embed_size"], key=key_cond)

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

    # ------------------------------------------------------------------
    #                       PUBLIC HIGH‑LEVEL API
    # ------------------------------------------------------------------

    def behead(self):
        # removes the glue and the decoder
        s = self
        s = eqx.tree_at(lambda x: x.decoder_glue, s, lambda inp: inp)
        s = eqx.tree_at(lambda x: x.decoder, s, lambda inp: inp)
        return s

    # ------------------------------------------------------------------
    #                         INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _build_causal_mask(self, seq_len: int) -> Bool[Array, "seq_len seq_len"]:
        return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

    def embed(self, xs, ys):
        """Encode (xs, ys) and pass through *causal* transformer blocks."""
        x = self.encoder(xs, ys)
        causal_mask = self._build_causal_mask(x.shape[0])
        for layer in self.layers:
            x = layer(x, causal_mask)  # causal mask shared across layers
        x = eqx.error_if(
            x,
            jnp.any(jnp.isnan(x)),
            "Nans encountered after the transformer layers",
        )
        return x

    # ------------------------------------------------------------------
    #                          MAIN FORWARD
    # ------------------------------------------------------------------

    def __call__(self, times, ys, target_time):
        embeddings = self.embed(times, ys)
        # contains the embeddings of all the subcurves
        # each subcurve _can_ be conditioned separately, but we ignore this case for now

        target_time = target_time.squeeze()
        assert (
            target_time.shape == ()
        ), "all subcurves should be conditioned on the same time (FIXME)"
        conditioned = eqx.filter_vmap(
            lambda single_embed: self.conditioner(single_embed, target_time)
        )(embeddings)
        histograms = eqx.filter_vmap(lambda x: self.decoder(self.decoder_glue(x)))(conditioned)
        return histograms

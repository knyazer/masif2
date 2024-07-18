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


class PFN(eqx.Module):
    layers: list[TransformerLayer]
    encoder: Encoder
    decoder_glue: eqx.nn.Linear
    decoder: Decoder

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
        assert decoder is not None
        assert key is not None
        # each transformer block wants a different key
        key_layers, key_glue = jr.split(key, 2)
        keys = jr.split(key_layers, n_layers)
        del key  # avoid shadowing
        self.encoder = encoder
        self.layers = [TransformerLayer(key=_key, **kws) for _key in keys]
        self.decoder_glue = eqx.nn.Linear(
            kws["embed_size"],
            decoder.n_bins,  # type: ignore
            key=key_glue,
        )
        self.decoder = decoder

    def __call__(self, xs, ys, mask, target_x):
        x, mask = self.encoder(xs, ys, mask, target_x)

        for layer in self.layers:
            x = layer(x, mask)
        latent = x[-1]  # take only the last token, which is the target one 'by design'

        x = eqx.error_if(
            x,
            jnp.any(jnp.isnan(x)),
            "Nans encountered after the transformer layers",
        )
        x = self.decoder_glue(latent)
        x = self.decoder(x)
        return x

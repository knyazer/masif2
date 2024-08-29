import equinox as eqx
import jax
from einops import repeat
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, PRNGKeyArray


class Encoder(eqx.Module):
    def __call__(self, *_, **__):
        raise NotImplementedError


class Embedder(eqx.Module):
    def __call__(self, *_, **__):
        raise NotImplementedError


class FourierEmbedder(Embedder):
    phases: Float[Array, "n"]
    frequencies: Float[Array, "n"]
    input_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)

    def __init__(self, input_size: int, output_size: int, key=None):
        assert key is not None  # TODO: support higher dimensions
        assert input_size == 1, "Unstable and probably broken for other input dims"
        self.input_size = input_size
        self.output_size = output_size

        freq_key, phase_key = jr.split(key, 2)
        self.frequencies = jr.uniform(
            freq_key,
            (input_size, output_size),
            minval=0.1,
            maxval=1000,
        )
        self.phases = jr.uniform(
            phase_key,
            (output_size,),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )

    def __call__(self, x: Float[Array, "n"]):
        assert (
            x.shape[0] == self.input_size
        ), f"fourier embedder input size mismatch {x.shape[0]}"
        freqs = jax.lax.stop_gradient(self.frequencies)
        phases = jax.lax.stop_gradient(self.phases)
        out = jnp.cos((x @ (1.0 / freqs)) + phases)
        assert out.shape == (self.output_size,), "Critical internal failure in fourier"
        return out


class LinearEmbedder(Embedder):
    layer: eqx.nn.Linear
    input_size: int = eqx.field(static=True)

    def __init__(self, input_size: int, output_size: int, key=None):
        assert key is not None
        self.layer = eqx.nn.Linear(input_size, output_size, key=key)
        self.input_size = input_size

    def __call__(self, x: Float[Array, "n"]):
        assert (
            x.shape[0] == self.input_size
        ), f"linear embedder input size mismatch {x.shape[0]}"
        return jax.nn.gelu(self.layer(x), approximate=True)


class JointEncoder(Encoder):
    value_embedder: Embedder
    position_embedder: Embedder
    learnable_target: Float[Array, "n"]

    def __init__(
        self,
        positional_embedding_size: int = 16,
        value_embedding_size: int = 16,
        embedding_type: str = "fourier",
        key: PRNGKeyArray | None = None,
    ):
        assert key is not None, "Pass the key to the encoder!"
        assert embedding_type == "fourier", "That was an illusion of choice"
        value_key, pos_key, lear_key = jr.split(key, 3)

        # each input is a pair (x, y) is mapped to a loong vector,
        # first half of which is x -> positional embedding, and second half
        # is y -> value embedding
        self.position_embedder = FourierEmbedder(
            1,
            positional_embedding_size,
            key=pos_key,
        )
        self.value_embedder = LinearEmbedder(
            1,
            value_embedding_size,
            key=value_key,
        )
        self.learnable_target = jr.normal(lear_key, (value_embedding_size,))

    def __call__(
        self,
        x: Float[Array, "n"],
        y: Float[Array, "n"],
        mask: Bool[Array, "n"],
        target_x: Float[Array, ""],
    ):  # TODO: lookup docs about how to deal with 2 * n
        # append a new token, for the target
        x = jnp.concatenate([x, jnp.array([target_x])], axis=0)
        mask = jnp.concatenate([mask, jnp.array([True])], axis=0)
        y = jnp.concatenate([y, jnp.array([0])], axis=0)

        # to avoid issues with adjoint becoming nan we replace all nans
        # with a magic value here (a zero, technically)
        x = jnp.nan_to_num(x)
        y = jnp.nan_to_num(y)

        pos_embedding = eqx.filter_vmap(self.position_embedder)(x[:, None])
        val_embedding = eqx.filter_vmap(self.value_embedder)(y[:, None])

        # mask the 'masked' values with zeroes
        pos_embedding: jax.Array = jnp.where(
            repeat(mask, f"n -> n {pos_embedding.shape[-1]}"),
            pos_embedding,
            jnp.zeros_like(pos_embedding),
        )
        val_embedding: jax.Array = jnp.where(
            repeat(mask, f"n -> n {val_embedding.shape[-1]}"),
            val_embedding,
            jnp.zeros_like(val_embedding),
        )
        val_embedding = val_embedding.at[-1].set(self.learnable_target)

        out = jnp.concatenate([pos_embedding, val_embedding], axis=-1)

        out = eqx.error_if(
            out,
            jnp.any(jnp.isnan(out)),
            "Encoder call resulted in nans",
        )

        reps = x.shape[0]
        mask_hor = repeat(mask, f"n -> n {reps}")
        mask_ver = repeat(mask, f"n -> {reps} n")
        attn_mask = jnp.bitwise_and(mask_hor, mask_ver)
        return out, attn_mask

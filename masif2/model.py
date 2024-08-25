from collections.abc import Callable

import equinox as eqx
from einops import repeat
from jaxtyping import Array, Bool, Float


class MASIF2(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    hyp_weighting: eqx.Module | Callable

    def __init__(self, /, encoder, decoder, hyp_weighting):
        self.encoder = encoder
        self.decoder = decoder
        self.weighting = hyp_weighting

    def __call__(
        self,
        xs: Float[Array, "n xdim"],
        ys: Float[Array, "n xdim"],
        masks: Bool[Array, "n xdim"],
        x_target: Float[Array, ""],
        hyps: Float[Array, "n hdim"],
        hyp_target: Float[Array, "hdim"],
    ):
        input_len = len(xs)
        assert len(ys) == input_len
        assert len(masks) == input_len
        print(f"Recompiling MASIF for the {input_len}")
        latents = eqx.filter_vmap(self.encoder)(
            xs,
            ys,
            masks,
            repeat(x_target, f"... -> {input_len} ..."),
        )
        weights = eqx.filter_vmap(eqx.Partial(self.weighting, target=hyp_target))(hyps)
        mixed_latent = latents * weights / weights.sum()

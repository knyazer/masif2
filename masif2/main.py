from os import wait
from typing import Any

import time
import einops
import equinox as eqx
import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.special import ndtri as normal_icdf
from jaxtyping import Array, Float, PRNGKeyArray
from equinox import internal as eqxi
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import os
import optax
import wandb
import functools

from masif2.pfn import PFN, HistogramDecoder, JointEncoder
from masif2.common import (
    eval_model,
    load_dataset,
    distance_weights,
    pad_to_ctx,
    make_batch,
    make_seed_from_key,
    train_folders,
)

T = 50


@functools.lru_cache
def load_model(model_name, kind):
    print(f"Loaded {model_name}...")
    sample_hypercube_hp = lambda k: jr.uniform(k, shape=(10,))
    pi_config = PiConfigSet(sample_hypercube_hp, jr.PRNGKey(0))
    model = MASIF(jr.PRNGKey(1), pi_config=pi_config, kind=kind)

    model = eqx.tree_deserialise_leaves(model_name, model)
    model = eqx.nn.inference_mode(model)
    return model


def pad_to_shape(arr, target_shape, pad_value=0):
    current_shape = arr.shape
    padding = [(0, max(t - s, 0)) for s, t in zip(current_shape, target_shape)]
    return jnp.pad(arr, padding, constant_values=pad_value)


class Normalizer(eqx.Module):
    """
    Takes a bunch of scalar-valued samples, and attempts to normalize them to U(0, 1),
    (this is used to reproduce ifbo behaviour, Appendix A1, if this is applied dimensionwise
    we get the correct marginals, but the joint distribution is not uniform)
    """

    inv_cdf: Float[Array, "m"] | None

    def __init__(self):
        self.inv_cdf = None

    def fit(self, data: Float[Array, "n"], bins=1000):
        # we just estimate the inverse CDF by sorting the data, and then
        # linearly interpolating between the bins
        sorted_data = jnp.sort(data)
        new_inv_cdf = jnp.interp(
            jnp.linspace(0, 1, bins), jnp.linspace(0, 1, len(sorted_data)), sorted_data
        )
        return eqx.tree_at(lambda s: s.inv_cdf, self, new_inv_cdf)

    def __call__(self, x):
        if self.inv_cdf is None:
            raise ValueError("Normalizer not fitted yet")
        # the reason why not (0,1) but (0.001, 0.999) is cuz we want icdf to be bounded
        return jnp.nan_to_num(
            jnp.interp(x, self.inv_cdf, jnp.linspace(0.0001, 0.9999, num=len(self.inv_cdf)))
        )


if __name__ == "__main__":
    # simple test
    data = jr.normal(jr.key(0), (1000,))
    normalizer = Normalizer().fit(data)
    out = normalizer(jnp.array([-3, 0, 3]))
    assert jnp.all(out >= 0)
    assert jnp.all(out <= 1)
    assert out[0] < 0.03
    assert abs(out[1] - 0.5) < 0.03
    assert out[2] > 0.97
    del data, normalizer, out


class RandomizedMLP(eqx.Module):
    linears: list  # `eqx.nn.Linear` layers
    preactivation_noise_std: Any
    output_noise: Any
    normalizers: Normalizer  # (stacked)

    def __init__(self, input_dim: int, key: PRNGKeyArray):
        k_arch, k_layers, k_misc, k_spar = jr.split(key, 4)

        # Hyperparameters borrowed from IFBO implementation
        num_layers = 6
        hidden_size = 100
        init_std = jr.uniform(k_arch, (), minval=0.089, maxval=0.193)
        sparsity = jr.uniform(k_spar, (), minval=0.05, maxval=0.8)
        self.preactivation_noise_std = jr.uniform(k_arch, (), minval=0.0003, maxval=0.0014)
        self.output_noise = jr.uniform(k_arch, (), minval=0.0004, maxval=0.0013)

        # ---- (2) make the linear stack with custom init & sparsity ----
        layer_keys = jr.split(k_layers, num_layers)
        init_keys = jr.split(k_misc, num_layers * 3)  # (weight, bias, mask) per layer
        linears = []
        ik = 0
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_size
            out_features = 26 if i == num_layers - 1 else hidden_size
            lin = eqx.nn.Linear(in_features, out_features, key=layer_keys[i])

            # custom weight/bias init ~ N(0, init_std²)
            w_key, b_key, m_key = init_keys[ik], init_keys[ik + 1], init_keys[ik + 2]
            ik += 3
            weight = jr.normal(w_key, lin.weight.shape) * init_std
            bias = jr.normal(b_key, lin.bias.shape) * init_std

            # apply sparsity to *hidden* layers only (not first, not last)
            if 0 < i < num_layers - 1 and sparsity > 0.0:
                mask = jr.bernoulli(m_key, p=1.0 - sparsity, shape=weight.shape)
                weight = weight * mask / jnp.sqrt(1.0 - sparsity)

            lin = eqx.tree_at(lambda l: (l.weight, l.bias), lin, (weight, bias))
            linears.append(lin)
        self.linears = linears

        # ---- (3) one normaliser per output dimension ----
        normalizers = [Normalizer() for _ in range(26)]
        self.normalizers = jax.tree.map(
            lambda *xs: jnp.stack(xs), *normalizers, is_leaf=eqx.is_array
        )

    # ------------------------- fit --------------------------------
    def fit(self, inputs: Float[Array, "batch input_dim"]):
        """Estimate inverse‑CDFs so that each output marginal is ≈U(0,1)."""
        outputs = eqx.filter_vmap(self.forward)(inputs)  # (batch, 26)
        new_norm = eqx.filter_vmap(lambda n, d: n.fit(d))(
            self.normalizers, jnp.swapaxes(outputs, 0, 1)
        )
        return eqx.tree_at(lambda s: s.normalizers, self, new_norm)

    def forward(self, x: Float[Array, "input_dim"], *, key: PRNGKeyArray | None = None):
        if x.ndim != 1:
            raise ValueError("`forward` expects a rank‑1 vector; vmap over samples.")

        # walk through hidden layers
        for idx, lin in enumerate(self.linears[:-1]):
            x = lin(x)
            # optional pre‑activation noise
            if key is not None:
                key, sub = jr.split(key)
                x = x + jr.normal(sub, x.shape) * self.preactivation_noise_std
            x = jax.nn.tanh(x)

        # final linear + output noise
        x = self.linears[-1](x)
        if key is not None:
            key, sub = jr.split(key)
            x = x + jr.normal(sub, x.shape) * self.output_noise
        return x

    def __call__(self, x: Float[Array, "input_dim"], *, key: PRNGKeyArray | None = None):
        raw = self.forward(x, key=key)  # (26,)
        out = eqx.filter_vmap(lambda n, y: n(y))(self.normalizers, raw)
        return out


if __name__ == "__main__":
    # quick test: check that the output is in [0, 1]
    data = jr.normal(jr.key(0), (1000, 6))
    data2 = jr.normal(jr.key(0), (1000, 6))
    mlp = RandomizedMLP(6, jr.key(0))
    mlp = mlp.fit(data)
    out = eqx.filter_vmap(mlp)(data2)

    assert jnp.all(out >= 0)
    assert jnp.all(out <= 1)
    assert jnp.any(out < 0.03)
    assert jnp.any(out > 0.97)
    s1 = jnp.sum(out[:, 1] < 0.1)
    s2 = jnp.sum(out[:, 4] > 0.9)
    s3 = jnp.sum((out[:, 4] > 0.45) & (out[:, 4] < 0.55))
    assert s1 < 120 and s1 > 80  # noqa
    assert s2 < 120 and s2 > 80  # noqa
    assert s3 < 120 and s3 > 80  # noqa

    del data, data2, mlp, out, s1, s2, s3


def compute_xt(t, xsat, rsat):
    xt = jnp.where(t <= xsat, t, rsat * (t - xsat) + xsat)
    return xt


def pow4(t, alpha, rsat, xsat, ysat, eps):
    xt = compute_xt(t, xsat, rsat)
    output = 1 - jnp.power((eps ** (-1 / alpha) - 1) * (xt / xsat) + 1, -alpha)
    return ysat * output


def exp4(t, alpha, rsat, xsat, ysat, eps):
    xt = compute_xt(t, xsat, rsat)
    output = 1 - jnp.power(eps, jnp.power(xt / xsat, alpha))
    return ysat * output


def ilog4(t, alpha, rsat, xsat, ysat, eps):
    xt = compute_xt(t, xsat, rsat)
    term1 = jnp.power(alpha, 1.0 / eps) - alpha
    term2 = xt / xsat
    output = 1 - jnp.log(alpha) / jnp.log(term1 * term2 + alpha)
    return ysat * output


def hill4(t, alpha, rsat, xsat, ysat, eps):
    xt = compute_xt(t, xsat, rsat)
    term1 = jnp.power(xt / xsat, alpha)
    output = 1 - jnp.power(term1 * (1 / eps - 1) + 1, -1)
    return ysat * output


class CurveHypers(eqx.Module):
    variance: Float[Array, ""]
    yinf: Float[Array, ""]
    y0: Float[Array, ""]
    w: Float[Array, "4"]
    rsats: Any
    xsats: Any
    ysats: Any
    epss: Any
    alphas: Any

    def __call__(self, t):
        # this is what is referred to as 'fcomb'
        s = jnp.array(0.0, dtype=jnp.float32)
        for i, fn in enumerate([pow4, exp4, ilog4, hill4]):
            fn_eval = fn(
                t, self.alphas[i], self.rsats[i], self.xsats[i], self.ysats[i], self.epss[i]
            )
            fn_eval = jnp.clip(fn_eval, 0, 1)
            s += self.w[i] * fn_eval
        return self.y0 + (self.yinf - self.y0) * s


class PiCurve(eqx.Module):
    hypers: Any

    def __init__(self, hypers: CurveHypers):
        self.hypers = hypers

    def __call__(self, t, key):
        assert t.size == 1

        # we return either the mean, if no key is specified
        # or the noised version, if the key is specified
        mean = self.hypers(t)

        if key is None:
            return mean
        return mean + jr.normal(key) * jnp.sqrt(self.hypers.variance)


class PiConfig(eqx.Module):
    """
    A sampler class for the pi_config: the mapping between intrinsic hyperspace,
    and the curve hyperspace.
    """

    mlp: Any
    indices: Any

    def __init__(self, lambda_gen: Any, key: PRNGKeyArray):
        """
        lambda_gen is a function that samples from a desired distribution of lambdas
        We need it instead of just a e.g. dimension of lambda because MLPs in IFBO
        are calibrated to have the "desired" marginal distribution (consult appendix A1)
        """
        k1, k2, k3 = jr.split(key, 3)
        _lambda = lambda_gen(k1)
        assert len(_lambda.shape) == 1, "output of lambda_gen should be a one-dim vector"

        self.mlp = RandomizedMLP(input_dim=_lambda.shape[0], key=k2)

        # fit estimates the inverse CDF of the mlp, and makes the mlp output have
        # dimensionwise/marginally U(0,1) distribution
        self.mlp = self.mlp.fit(eqx.filter_vmap(lambda_gen)(jr.split(k3, 10_000)))

        self.indices = jnp.arange(26)

    def __call__(self, _lambda):
        # Takes lambda (variable name) as input, returns a hyper, which allows to sample the curves
        assert _lambda.ndim == 1, (
            f"probs forgot to vmap the call to pi config? lambda shape was {_lambda.shape}"
        )

        out = self.mlp(_lambda)[self.indices]

        assert len(out) == len(self.indices)

        # guessing these params from the Table2 label
        u1 = out[19]
        u2 = out[20]
        u3 = out[21]
        assert u1.size == 1
        assert u2.size == 1
        ymax = jax.lax.cond(u3 < 0.25, lambda: jnp.max(jnp.array([u1, u2])), lambda: jnp.array(1.0))
        y0 = jnp.min(jnp.array([u1, u2]))
        yinf = out[18] * (ymax - y0) + y0

        # Table 2, A1
        alpha1 = jnp.exp(normal_icdf(out[1]) * 1 + 1)
        alpha2 = jnp.exp(normal_icdf(out[2]) * 1 + 0)
        alpha3 = jnp.exp(normal_icdf(out[3]) * 1 - 4) + 1
        alpha4 = jnp.exp(normal_icdf(out[4]) * 0.5 + 0.5)
        alphas = jnp.vstack([alpha1, alpha2, alpha3, alpha4]).ravel()
        assert alphas.shape == (4,)

        xsats = jnp.exp(normal_icdf(out[5:9]))
        assert len(xsats) == 4

        epss = jnp.exp(out[9:13] * -3)
        assert len(epss) == 4

        ysats = yinf - epss * (yinf - y0)
        rsats = jnp.exp(1 - jnp.exp(out[22:26]))

        # weights are sampled (seemingly) directly from the model, thus having U[0,1] distr
        weights = out[13:17]
        weights = weights / jnp.sum(weights)
        variance = jnp.exp(normal_icdf(out[17]) - 5) ** 2
        # it think they also sample t-something, which has a reference to in 4.2, but
        # since i don't get how to apply it - i don't think i care
        # _at least for now_
        return PiCurve(
            CurveHypers(
                variance=variance,
                yinf=yinf,
                y0=y0,
                alphas=alphas,
                epss=epss,
                xsats=xsats,
                ysats=ysats,
                rsats=rsats,
                w=weights,
            )
        )

    def permute(self, key):
        return eqx.tree_at(lambda x: x.indices, self, jr.permutation(key, len(self.indices)))

    def make(self, _lambda, key):
        curve_gen = self.__call__(_lambda)
        if key is not None:
            curve = eqx.filter_vmap(curve_gen)(
                (jnp.arange(T).astype(jnp.float32) / T)[:, None], jr.split(key, T)
            )
        else:
            curve = eqx.filter_vmap(lambda x: curve_gen(x, key=None))(
                (jnp.arange(T).astype(jnp.float32) / T)[:, None]
            )
        return curve


def soa_to_aos(soa, size):
    return [
        jax.tree.map(lambda x: x[i] if eqx.is_array(x) and x.shape[0] == size else x, soa)
        for i in range(size)
    ]


class PiConfigSet(eqx.Module):
    """
    just a little wrapper that generates a fixed number (e.g. 100) of pi configs, and chooses
    one of them at runtime arbitrarily (based on a key).

    The reason to have a pre-set number of pi configs is so that we don't spend too much time
    calibrating them (each pi config makes an mlp, and then ensures the marginals are uniform,
    which takes quite a bit of time)
    """

    configs: list
    N: int

    def __init__(self, lambda_gen, key, *, N=500):
        self.N = N
        configs_aos = [PiConfig(lambda_gen, _key) for _key in jr.split(key, self.N)]

        self.configs = jax.tree.map(
            lambda *args: jnp.stack(args), *configs_aos, is_leaf=eqx.is_array
        )

    def get_config(self, key):
        k1, k2 = jr.split(key)
        index = jr.randint(k1, shape=(), minval=0, maxval=self.N - 1)
        pi_config = jax.tree.map(lambda x: x[index], self.configs, is_leaf=eqx.is_array).permute(k2)
        return pi_config


n_hyps = 10
n_curves = 8


class HyperMap(eqx.Module):
    embed: Any
    embed2: Any

    def __init__(self, key, embed_dim=0):
        k1, k2, k3 = jr.split(key, 3)
        self.embed = eqx.nn.Linear(10 + embed_dim, 32, key=k1)
        self.embed2 = eqx.nn.Linear(32, 32, key=k2)

    def __call__(self, x, latent=None):
        if latent is None:
            x = jax.nn.gelu(self.embed(x))
        else:
            x = jax.nn.gelu(self.embed(jnp.concatenate([x, latent], axis=0)))
        x = jax.nn.gelu(self.embed2(x))
        return x


class MASIF(eqx.Module):
    encoder: PFN
    decoder: Any  # the (interpolated embedding, time(?) -> histogram) decoder
    hyper_map: Any
    hyper_map_with_embeds: Any
    glue: Any
    inv_cov_prm: Any
    cmethod: Any
    Q: Any
    K: Any
    proj: Any

    def __init__(self, key, pi_config=None, kind=None, quick=False):
        assert kind is not None
        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = jr.split(key, 10)
        embedder = JointEncoder(key=k1)
        self.encoder = PFN(
            encoder=embedder,
            n_layers=8,
            decoder=None,
            key=k2,
            hidden_size=64,
            embed_size=64,
            num_heads=4,
        )
        self.hyper_map = HyperMap(k5)
        self.hyper_map_with_embeds = HyperMap(k6, embed_dim=64)

        self.Q = eqx.nn.Linear(32, 32, key=k8)
        self.K = eqx.nn.Linear(32, 32, key=k9)
        self.proj = eqx.nn.Linear(64, 1, key=k10)

        self.cmethod = kind

        self.glue = eqx.nn.Linear(64, 500, key=k5)
        self.decoder = HistogramDecoder(n_bins=500)

        sample_hypercube_hp = lambda key: jr.uniform(key, shape=(n_hyps,))
        if pi_config is None:
            pi_config = PiConfigSet(sample_hypercube_hp, k6)
        if pi_config is not None and pi_config == False:  # noqa
            self.decoder = None
            self.inv_cov_prm = None
            return

        n_curves = 5_000 if not quick else 10
        curves = eqx.filter_vmap(
            lambda key: pi_config.get_config(key)(sample_hypercube_hp(jr.split(key)[0]))
        )(jr.split(k3, n_curves))

        points_to_fit = eqx.filter_vmap(lambda c, t, k: c(t, k))(
            curves, jr.uniform(k7, (n_curves,)), jr.split(k4, n_curves)
        )
        self.decoder = self.decoder.fit(points_to_fit)

        self.inv_cov_prm = jnp.eye(n_hyps).astype(jnp.float32)

    def generate_embeddings(self, curves, lengths=None):
        """
        Return per-point embeddings for a batch of (possibly padded) curves.
        For any padded points (i ≥ length[k] for curve k) the embedding is
        forced to equal the *last valid* embedding of that curve.
        """
        curves = curves[..., 0]  # (#curves, #points)
        if lengths is None:
            lengths = jnp.array([curves.shape[-1]] * curves.shape[0])

        assert curves.ndim == 2, f"curves should be (n_curves, n_points), got {curves.shape}"

        xs = jnp.arange(curves.shape[-1])
        xs = einops.repeat(xs, f"num_points -> {curves.shape[0]} num_points").astype(jnp.float32)

        embeddings = eqx.filter_vmap(  # (#curves, #points, d)
            lambda t, c: self.encoder.embed(t, c)
        )(xs, curves)

        num_curves, num_points, _ = embeddings.shape
        lengths = lengths.astype(jnp.int32)

        last_idx = lengths - 1  # (n_curves,)
        last_embed = embeddings[jnp.arange(num_curves), last_idx]  # (n_curves, d)

        mask = jnp.arange(num_points)[None, :] >= lengths[:, None]  # (n_curves, n_points)
        embeddings = jnp.where(
            mask[..., None],  # broadcast to (n_curves,n_points,1)
            last_embed[:, None, :],  # broadcast to (n_curves,1,d)
            embeddings,
        )

        return embeddings  # (#curves, #points, d)

    def make_embedding_combinations(self, encoded_curves, num_combinations, key):
        num_curves, points_per_curve, hidden_dim = encoded_curves.shape
        valid_mask = ~(jnp.isnan(encoded_curves) | jnp.isinf(encoded_curves))
        valid_mask = jnp.any(valid_mask, axis=-1)
        point_indices = []
        for idx in range(
            num_curves
        ):  # we assume there are not enough curves to make compile time of a for-loop a problem
            key, subkey = jr.split(key)
            curve_points = jr.choice(
                subkey,
                encoded_curves[idx],
                p=valid_mask[idx],  # sample only valid points
                shape=(num_combinations,),
                replace=True,  # with replacement in case there is e.g. only 1 valid point
            )
            point_indices.append(curve_points)
        out = jnp.stack(point_indices)
        return einops.rearrange(out, "hp reps latent -> reps hp latent")

    def combine_embeddings(self, embeddings, hypers, target_hyper, mask=None):
        if hypers.shape[1] > n_hyps:
            raise RuntimeError(
                f"yikes, the passed hypers were of shapes {hypers.shape} -> "
                f"{target_hyper.shape}, while the maximum allowed shape (the "
                f"size of covariance) is {len(self.inv_cov_prm)}"
            )
        if mask is None:
            mask = jnp.ones((len(embeddings),))

        cov_len = len(self.inv_cov_prm)
        if cov_len > len(hypers):
            hypers = pad_to_shape(hypers, (len(hypers), cov_len))
            target_hyper = pad_to_shape(target_hyper, (cov_len,))

        diff = hypers - target_hyper
        # diff = eqx.error_if(diff, jnp.any(jnp.isnan(diff)), "diffs are nans")
        # embeddings = eqx.error_if(embeddings, jnp.any(jnp.isnan(embeddings)), "nans in embeddings")

        if self.cmethod == "identity":
            sq_dists = jnp.einsum("ij,ik->i", diff, diff)  # cute trick to get diag
            dists = jnp.sqrt(sq_dists + 1e-7)  # roots ofc
            weights = 1.0 / (1e-7 + dists)
        elif self.cmethod == "covariance":
            # make the inverse covariance (it must be SPD)
            tri = jnp.tril(self.inv_cov_prm)
            inv_cov = (tri @ tri.T) + jnp.eye(len(self.inv_cov_prm)) * 1e-7
            # inv_cov = eqx.error_if(
            #    inv_cov, jnp.any(jnp.isnan(self.inv_cov_prm)), "nans in covariance prm"
            # )
            # inv_cov = eqx.error_if(
            #    inv_cov, jnp.any(jnp.isnan(inv_cov)), "nans in inverse covariance"
            # )
            sq_dists = jnp.sum(
                jnp.nan_to_num(diff) * (jnp.nan_to_num(diff) @ jnp.nan_to_num(inv_cov)), axis=1
            )
            dists = jnp.sqrt(sq_dists + 1e-7)  # roots ofc
            # dists = eqx.error_if(dists, jnp.any(jnp.isnan(dists)), "nans in dists")
            weights = 1.0 / (1e-7 + dists)
        elif self.cmethod == "learned":
            target_hyper_latent = self.hyper_map(target_hyper)
            hypers_latents = eqx.filter_vmap(self.hyper_map_with_embeds)(hypers, embeddings)
            weights = jnp.exp(
                eqx.filter_vmap(lambda x: (self.K(x) * self.Q(target_hyper_latent)).sum())(
                    hypers_latents
                )
            )
        elif self.cmethod == "learned_nolatent":
            target_hyper_latent = self.hyper_map(target_hyper)
            hypers_latents = eqx.filter_vmap(self.hyper_map)(hypers)
            weights = jnp.exp(
                eqx.filter_vmap(lambda x: (self.K(x) * self.Q(target_hyper_latent)).sum())(
                    hypers_latents
                )
            )
        else:
            raise RuntimeError(f"combination method {self.cmethod} is not defined")

        weights = weights * mask

        weights = weights / jnp.sum(weights)  # norm
        # weights = eqx.error_if(weights, jnp.any(weights < 0), "weights are negative")
        # weights = eqx.error_if(weights, jnp.any(jnp.isnan(weights)), "weights are nans")
        assert weights.size == len(hypers)
        out = (embeddings * weights[:, None]).sum(axis=0)
        # out = eqx.error_if(out, jnp.any(jnp.isnan(out)), "out of combination of embeddings is nans")
        assert out.size == embeddings[0].size
        return out

    def get_borders(self):
        borders = jnp.concatenate([jnp.array([0.0]), self.decoder.bounds[1:-1], jnp.array([1.0])])
        return borders

    def eval(
        self,
        curve_hypers,
        curves,
        curve_cutoffs,
        target_hyper,
        target_x,
        target_y,
        curve_mask=None,
        has_aux=True,
    ):
        assert curves.shape[0] == curve_cutoffs.shape[0]
        assert curves.shape[0] == curve_hypers.shape[0]
        curve_embeds_raw = self.generate_embeddings(curves, curve_cutoffs)

        # curve_embeds_raw = eqx.error_if(
        #    curve_embeds_raw, jnp.any(jnp.isnan(curve_embeds_raw)), "nans in embeds"
        # )
        curve_embeds = eqx.filter_vmap(
            lambda embed_single_curve: eqx.filter_vmap(
                lambda embed_single: self.encoder.conditioner(embed_single, target_x[0])
            )(embed_single_curve)
        )(curve_embeds_raw)

        # curve_embeds = eqx.error_if(
        #    curve_embeds, jnp.any(jnp.isnan(curve_embeds)), "nans in embeds"
        # )

        embeds = curve_embeds[jnp.arange(curve_cutoffs.shape[0]), -1]  # get the last embedding
        combined = self.combine_embeddings(embeds, curve_hypers, target_hyper, mask=curve_mask)

        weights = self.glue(combined)

        # predict from each combined
        hist = self.decoder(weights)
        pdfs = hist.pdf(target_y[0])
        # pdfs = eqx.error_if(pdfs, jnp.any(jnp.isnan(pdfs)), "pdfs are nans")

        ll = jnp.mean(jnp.log(pdfs))

        if has_aux:
            return ll, hist.repr()
        else:
            return ll

    def loss(self, curve_hypers, curves, target_hyper, target_xs, target_ys, key, lengths=None):
        key, subkey = jr.split(key)
        if len(target_ys) >= 2:
            target_ys = target_ys[:, 0]
        assert len(target_ys.shape) == 1
        num_combs = 50
        curve_embeds_raw = self.generate_embeddings(curves, lengths)

        # while doing such a conditiniong is a waste of compute: most of the embeddings
        # are not used anyways; it is a pretty small waste of compute, and, overall
        # it is easier to reason about this setup, so i prefer it
        curve_embeds = eqx.filter_vmap(
            lambda t: eqx.filter_vmap(
                lambda embed_single_curve: eqx.filter_vmap(
                    lambda embed_single: self.encoder.conditioner(embed_single, t)
                )(embed_single_curve)
            )(curve_embeds_raw)
        )(target_xs)

        # curve_embeds is now (#target times, #hyps, #n curve, #embedding), oof
        # I shall vmap the combinations for each target time
        # Also, we are fine with not preserving the same hyp order
        # so the rest is one big vmap
        def subseq_curves(curve_embeds, target_y, subkey):
            combinations = self.make_embedding_combinations(curve_embeds, num_combs, subkey)

            # combinations = eqx.error_if(
            #    combinations, jnp.any(jnp.isnan(combinations)), "nans after combinations"
            # )

            # for each combination: combine it
            combined = eqx.filter_vmap(
                lambda comb: self.combine_embeddings(comb, curve_hypers, target_hyper)
            )(combinations)

            # combined = eqx.error_if(
            #    combined, jnp.any(jnp.isnan(combined)), "nans after combine embeddings"
            # )

            weights = eqx.filter_vmap(self.glue)(combined)

            # predict from each combined
            histograms = eqx.filter_vmap(lambda x: self.decoder(x))(weights)
            pdfs = eqx.filter_vmap(lambda hist: hist.pdf(target_y))(histograms)
            # pdfs = eqx.error_if(pdfs, jnp.any(jnp.isnan(pdfs)), "pdfs are nans")
            assert pdfs.size == num_combs

            return jnp.mean(jnp.log(pdfs))

        out = eqx.filter_vmap(subseq_curves)(
            curve_embeds, target_ys, jr.split(subkey, curve_embeds.shape[0])
        ).mean()

        return out


def full_train(kind, seed=0, model_name=None):
    if model_name is None:
        model_name = f"masif_{model_name}.eqx"

    key = jr.key(seed)
    k1, k2, k3 = jr.split(key, 3)
    sample_hypercube_hp = lambda key: jr.uniform(key, shape=(10,))
    pi_config = PiConfigSet(sample_hypercube_hp, k2)
    masif = MASIF(k1, pi_config=pi_config, kind=kind)

    def train_step(model: MASIF, key):
        k1, k2, k3, k4, k5, k6, k7, k8 = jr.split(key, 8)
        curve_maker = pi_config.get_config(k4)

        _lambdas = jr.uniform(k1, (n_curves, n_hyps), minval=0.0, maxval=1.0)
        target_lambda = jr.uniform(k3, (n_hyps,), minval=0.0, maxval=1.0)
        rho = jr.uniform(k8, minval=0.0, maxval=1.0)
        _lambdas = target_lambda * rho + _lambdas * (1 - rho)

        subspace_size = jr.randint(k7, (), 1, n_hyps + 1)

        mask = (jnp.arange(n_hyps) < subspace_size).astype(jnp.int32)
        basis = jr.normal(k6, (n_hyps, n_hyps))
        basis = basis / jnp.linalg.norm(basis, axis=1)
        proj_fn = lambda x: jnp.clip((((basis @ x) + 1) / 2) * mask, 1e-6, 1.0 - 1e-5)
        target_lambda = proj_fn(target_lambda)
        _lambdas = jax.vmap(proj_fn)(_lambdas)

        _lambdas = jnp.clip(_lambdas, 0, 1)
        target_lambda = jnp.clip(target_lambda, 0, 1)

        # _lambdas = eqx.error_if(_lambdas, jnp.any(jnp.isnan(_lambdas)), "woof")
        # target_lambda = eqx.error_if(target_lambda, jnp.any(jnp.isnan(target_lambda)), "meow")

        curves = eqx.filter_vmap(curve_maker.make)(_lambdas, jr.split(k2, len(_lambdas)))
        # curves = eqx.error_if(curves, jnp.any(jnp.isnan(curves)), "moo")

        n_targets = 5

        k_targets, _ = jr.split(k5)

        def make_targets(key):
            """
            Sample `n_targets` distinct x–locations and fetch the noiseless y values.
            """
            xs = jr.choice(
                key,
                jnp.arange(T),  # candidates
                shape=(n_targets,),  # we want n_targets indices back
                replace=False,  # <- **without** replacement
            )

            # 2. evaluate the curve once and slice out the chosen points
            ys = curve_maker.make(target_lambda, None)[xs]
            return xs, ys

        target_xs, target_ys = make_targets(k_targets)

        # target_xs = eqx.error_if(target_xs, jnp.any(jnp.isnan(target_xs)), "kukareku")
        # target_ys = eqx.error_if(target_ys, jnp.any(jnp.isnan(target_ys)), "kukareku but for ys")

        losses = model.loss(_lambdas, curves, target_lambda, target_xs, target_ys, key=k4)
        return -losses.mean()

    @eqx.filter_jit
    def train_loss(model, key, batch_size=1000):
        return eqx.filter_vmap(lambda k: train_step(model, k))(jr.split(key, batch_size)).mean()

    def step(model, opt_state, key):
        loss, grads = eqx.filter_value_and_grad(train_loss)(model, key)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        old_bounds = model.decoder.bounds
        model = eqx.apply_updates(model, updates)
        # the following is a dirty trick that i need to fix later... FIXME
        model = eqx.tree_at(lambda m: m.decoder.bounds, model, old_bounds)
        return model, opt_state, loss

    num_steps = 3000
    schedule = optax.cosine_decay_schedule(
        init_value=1.5e-3,
        decay_steps=num_steps,
        alpha=0.2,
    )
    optim = optax.apply_if_finite(
        optax.chain(optax.adamw(learning_rate=schedule, weight_decay=1e-5), optax.clip(1.0)),
        2,
    )
    opt_state = optim.init(eqx.filter(masif, eqx.is_inexact_array))

    wandb.init(project="masif2", name=model_name)
    for i in tqdm(range(num_steps)):
        key, subkey = jr.split(key, 2)
        masif, opt_state, loss = eqx.filter_jit(step)(masif, opt_state, subkey)
        wandb.log({"loss": loss})
        if i % 200 == 199:
            wandb.log({"full_eval": eval_model(masif, shortened=True, num_allocations=10)[0]})
            eqx.tree_serialise_leaves(model_name, masif)


def finetune(model_kind, benchmark, tuning_kind, num_curves_in_total):
    model_name = f"{model_kind}.eqx"
    model = load_model("models/masif_" + model_name, model_kind)
    out_dir = f"models/finetuned/{benchmark}/{model_kind}/{tuning_kind}/{num_curves_in_total}"
    out_path = f"{out_dir}/model.eqx"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def filter_trainable(model):
        trainable_part = eqx.filter(model, eqx.is_inexact_array)
        if tuning_kind == "comb":
            trainable_part = jax.tree.map_with_path(
                lambda path, leaf: None if path[0].name == "encoder" else leaf,
                trainable_part,
            )
            trainable_part = jax.tree.map_with_path(
                lambda path, leaf: None if path[0].name == "decoder" else leaf,
                trainable_part,
            )
            trainable_part = jax.tree.map_with_path(
                lambda path, leaf: None if path[0].name == "glue" else leaf,
                trainable_part,
            )
        return trainable_part

    folders = train_folders(benchmark)

    num_curves_per_subset = num_curves_in_total // len(folders)
    print(f"Using {num_curves_per_subset} curves per subset")
    data = []
    for ds_path in tqdm(folders, desc="Loading all the datasets for finetuning..."):
        train_ds = load_dataset(f"{benchmark}/{ds_path}")
        rng = np.random.default_rng(abs(hash(ds_path)))
        curve_indices = rng.choice(len(train_ds), size=(num_curves_per_subset,), replace=False)
        train_ds = [train_ds[idx] for idx in curve_indices]
        data.append(train_ds)

    num_subsets = len(data)
    val_subsets = num_subsets // 2
    n_train_subsets = num_subsets - val_subsets
    train_data = data[:n_train_subsets]
    val_data = data[n_train_subsets:]

    def step_fn(model, opt_state, inp):
        loss, grads = eqx.filter_value_and_grad(
            lambda m, inp: -eqx.filter_vmap(eqx.Partial(m.eval, has_aux=False))(*inp).mean()
        )(model, inp)
        updates, opt_state = optim.update(
            filter_trainable(grads), opt_state, filter_trainable(model)
        )
        # the following is a dirty trick that i need to fix later... FIXME
        old_bounds = model.decoder.bounds
        model = eqx.apply_updates(model, updates)
        model = eqx.tree_at(lambda m: m.decoder.bounds, model, old_bounds)
        return model, opt_state, loss

    key = jr.key(0)
    num_steps = 2000
    peak_lr = 2e-3 if tuning_kind == "comb" else 2e-4
    batch_size = 2000 if tuning_kind == "comb" else 800
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=50,
        decay_steps=2 * num_steps // 3,
        end_value=peak_lr * 0.01,
    )
    optim = optax.apply_if_finite(
        optax.chain(optax.adamw(learning_rate=schedule, weight_decay=1e-6), optax.clip(1.0)),
        2,
    )
    opt_state = optim.init(filter_trainable(model))
    wandb.init(
        project="masif2", name=f"{model_kind}_{num_curves_in_total}_{benchmark}_{tuning_kind}"
    )
    val_loss, best_val_loss = 1e3, 1e3
    best_model = None
    all_val_inputs = []
    ekey, key = jr.split(key)
    for sz in [200, 400, 800]:
        for ds_val in val_data:
            esubkey, ekey = jr.split(ekey)
            all_val_inputs.append(
                make_batch(
                    seed=make_seed_from_key(esubkey),
                    size=500,
                    data=ds_val,
                    context_points=sz,
                )
            )

    for i in (pbar := tqdm(range(num_steps))):
        key, step_key, data_key = jr.split(key, 3)
        mx = jr.randint(step_key, (), minval=1, maxval=10) * 2
        inputs = make_batch(
            seed=make_seed_from_key(data_key),
            size=batch_size * 2 // (1 + (mx // 4)),
            data=train_data,
            context_points=100 * mx,
            wrapped=True,
        )

        model, opt_state, loss = eqx.filter_jit(step_fn)(model, opt_state, inputs)

        if i % 10 == 0:
            val_loss = 0.0
            for inp in tqdm(all_val_inputs):
                lls, _ = eqx.filter_vmap(model.eval)(*inp)
                val_loss += -lls.mean()
            val_loss = val_loss / len(all_val_inputs)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best loss achieved: {best_val_loss}, dumping the model to {out_path}")
                eqx.tree_serialise_leaves(out_path, model)

        pbar.set_description(f"Loss: {loss:.2f}, val_loss: {val_loss:.2f}")
        wandb.log({"loss": loss, "val_loss": val_loss})
    wandb.finish()


if __name__ == "__main__":
    for benchmark in ["taskset"]:  # ["lcbench", "taskset", "pd1"]
        for model in ["covariance"]:  # ["learned", "covariance"]
            for tuning_kind in ["comb"]:  # ["full", "comb"]
                for num_curves_in_total in [800, 1600]:
                    finetune(model, benchmark, tuning_kind, num_curves_in_total)

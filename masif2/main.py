from os import wait
from typing import Any

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
import optax
import wandb

from masif2.pfn import PFN, HistogramDecoder, JointEncoder


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
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    l3: eqx.nn.Linear
    normalizers: Normalizer

    def __init__(self, input_dim, key):
        k1, k2, k3 = jr.split(key, 3)
        self.l1 = eqx.nn.Linear(input_dim, 10, key=k1)
        self.l2 = eqx.nn.Linear(10, 20, key=k2)
        self.l3 = eqx.nn.Linear(20, 26, key=k3)
        # It is not clear whether the "marginals normalization" happens over all possible MLPs,
        # or over task-specific MLPs; but the former seems more 'adequate', and A1 implies that
        self.normalizers = [Normalizer() for _ in range(26)]
        self.normalizers = jax.tree.map(
            lambda *args: jnp.stack(args), *self.normalizers, is_leaf=eqx.is_array
        )

    def fit(self, inputs: Float[Array, "repeats input"]):
        outputs = eqx.filter_vmap(self.forward)(inputs)
        new_normalizers = eqx.filter_vmap(lambda n, data: n.fit(data))(
            self.normalizers, jnp.swapaxes(outputs, 0, 1)
        )
        return eqx.tree_at(lambda s: s.normalizers, self, new_normalizers)

    def forward(self, x):
        x = jax.nn.gelu(self.l1(x))
        x = jax.nn.gelu(self.l2(x))
        return self.l3(x)

    def __call__(self, x):
        if x.ndim != 1:
            raise ValueError("Only single samples supported: probably you forgot to vmap")
        out = eqx.filter_vmap(lambda n, x: n(x))(self.normalizers, self.forward(x))
        return jnp.clip(out, 0, 1)


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
            s = eqx.error_if(s, jnp.any(jnp.isnan(s)), "nans in fcomb, bad")
            s = eqx.error_if(
                s, jnp.any(s > 1), "weigheted sum of funcs could not be larger than 1 (consult A1)"
            )
            s = eqx.error_if(
                s, jnp.any(s < 0), "weigheted sum of funcs could not be smaller than 0 (consult A1)"
            )
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
    r"""
    A sampler class for the \pi_config: the mapping between intrinsic hyperspace,
    and the curve hyperspace.
    """

    mlp: Any

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
        self.mlp = self.mlp.fit(eqx.filter_vmap(lambda_gen)(jr.split(k3, 1000)))

    def __call__(self, _lambda):
        # Takes lambda (variable name) as input, returns a hyper, which allows to sample the curves
        assert (
            _lambda.ndim == 1
        ), f"probs forgot to vmap the call to pi config? lambda shape was {_lambda.shape}"

        out = self.mlp(_lambda)

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

    def __init__(self, lambda_gen, key, *, N=100):
        self.N = N
        configs_aos = [PiConfig(lambda_gen, _key) for _key in jr.split(key, self.N)]
        self.configs = jax.tree.map(
            lambda *args: jnp.stack(args), *configs_aos, is_leaf=eqx.is_array
        )

    def get_config(self, key):
        index = jr.randint(key, shape=(), minval=0, maxval=self.N)
        pi_config = jax.tree.map(lambda x: x[index], self.configs, is_leaf=eqx.is_array)
        return pi_config


n_hyps = 10
n_curves = 8


class HyperMap(eqx.Module):
    embed: Any
    embed2: Any
    embed3: Any

    def __init__(self, key):
        k1, k2, k3 = jr.split(key, 3)
        self.embed = eqx.nn.Linear(10, 32, key=k1)
        self.embed2 = eqx.nn.Linear(32, 32, key=k2)
        self.embed3 = eqx.nn.Linear(32, 64, key=k3)

    def __call__(self, x):
        x = jax.nn.gelu(self.embed(x))
        x = jax.nn.gelu(self.embed2(x)) + x
        x = jax.nn.gelu(self.embed3(x))
        return x


class MASIF(eqx.Module):
    encoder: PFN
    decoder: eqx.Module  # the (interpolated embedding, time(?) -> histogram) decoder
    hyper_map: eqx.Module
    glue: Any
    inv_cov_prm: Any
    cmethod: Any
    Q: Any
    K: Any
    proj: Any

    def __init__(self, key, pi_config=None):
        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = jr.split(key, 10)
        embedder = JointEncoder(key=k1)
        self.encoder = PFN(
            encoder=embedder,
            n_layers=6,
            decoder=None,
            key=k2,
            hidden_size=64,
            embed_size=64,
            num_heads=3,
        )
        self.hyper_map = HyperMap(k10)

        self.Q = eqx.nn.Linear(64, 64, key=k8)
        self.K = eqx.nn.Linear(64, 64, key=k9)
        self.proj = eqx.nn.Linear(64, 1, key=k10)

        self.cmethod = "learned"

        self.glue = eqx.nn.Linear(64, 500, key=k5)
        self.decoder = HistogramDecoder(
            n_bins=500
        )  # DO NOT CHANGE, STOPS BACKPROP FOR LARGER INDICES

        sample_hypercube_hp = lambda key: jr.uniform(key, shape=(n_hyps,))
        if pi_config is None:
            pi_config = PiConfigSet(sample_hypercube_hp, k6)
        if pi_config is not None and pi_config == False:  # noqa
            self.decoder = None
            self.inv_cov_prm = None
            return

        curves = eqx.filter_vmap(
            lambda key: pi_config.get_config(key)(sample_hypercube_hp(jr.split(key)[0]))
        )(jr.split(k3, 1_000))

        points_to_fit = eqx.filter_vmap(lambda c, t, k: c(t, k))(
            curves, jr.uniform(k7, (1_000,)), jr.split(k4, 1_000)
        )
        self.decoder = self.decoder.fit(points_to_fit)
        print(f"Bounds are {self.decoder.bounds}")

        self.inv_cov_prm = jnp.eye(n_hyps).astype(jnp.float32)

    def generate_embeddings(self, curves):
        # this function returns the embeddings for each curve point
        # for easier parallelization curves should be padded to max length
        curves = curves[..., 0]
        assert (
            len(curves.shape) == 2
        ), f"curves should be num_curves x num_points, got {curves.shape}"

        xs = jnp.arange(curves.shape[-1])
        xs = einops.repeat(xs, f"num_points -> {curves.shape[0]} num_points")
        xs = xs.astype(jnp.float32)
        assert xs.shape == curves.shape

        embeddings = eqx.filter_vmap(
            lambda time_range, curve: self.encoder.embed(time_range, curve)
        )(xs, curves)

        # generates embeddings of shape (#hyps, #n, #embed size)
        assert embeddings.shape[:2] == curves.shape
        return embeddings

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

    def combine_embeddings(self, embeddings, hypers, target_hyper):
        if hypers.shape[1] > n_hyps:
            raise RuntimeError(
                f"yikes, the passed hypers were of shapes {hypers.shape} -> "
                f"{target_hyper.shape}, while the maximum allowed shape (the "
                f"size of covariance) is {len(self.inv_cov_prm)}"
            )

        cov_len = len(self.inv_cov_prm)
        if cov_len > len(hypers):
            hypers = pad_to_shape(hypers, (len(hypers), cov_len))
            target_hyper = pad_to_shape(target_hyper, (cov_len,))

        diff = hypers - target_hyper
        diff = eqx.error_if(diff, jnp.any(jnp.isnan(diff)), "diffs are nans")
        embeddings = eqx.error_if(embeddings, jnp.any(jnp.isnan(embeddings)), "nans in embeddings")

        if self.cmethod == "identity":
            sq_dists = jnp.einsum("ij,ik->i", diff, diff)  # cute trick to get diag
            dists = jnp.sqrt(sq_dists + 1e-7)  # roots ofc
            weights = 1.0 / (1e-7 + dists)
        elif self.cmethod == "covariance":
            # make the inverse covariance (it must be SPD)
            tri = jnp.tril(self.inv_cov_prm)
            inv_cov = (tri @ tri.T) + jnp.eye(len(self.inv_cov_prm)) * 1e-7
            inv_cov = eqx.error_if(
                inv_cov, jnp.any(jnp.isnan(self.inv_cov_prm)), "nans in covariance prm"
            )
            inv_cov = eqx.error_if(
                inv_cov, jnp.any(jnp.isnan(inv_cov)), "nans in inverse covariance"
            )
            sq_dists = jnp.sum(
                jnp.nan_to_num(diff) * (jnp.nan_to_num(diff) @ jnp.nan_to_num(inv_cov)), axis=1
            )
            dists = jnp.sqrt(sq_dists + 1e-7)  # roots ofc
            dists = eqx.error_if(dists, jnp.any(jnp.isnan(dists)), "nans in dists")
            weights = 1.0 / (1e-7 + dists)
        elif self.cmethod == "learned":
            target_hyper_latent = self.hyper_map(target_hyper)
            hypers_latents = eqx.filter_vmap(self.hyper_map)(hypers)
            weights = jnp.exp(
                eqx.filter_vmap(lambda x: (self.K(x) * self.Q(target_hyper_latent)).sum())(
                    hypers_latents
                )
            )
        else:
            raise RuntimeError(f"combination method {self.cmethod} is not defined")

        weights = weights / jnp.sum(weights)  # norm
        weights = eqx.error_if(weights, jnp.any(weights < 0), "weights are negative")
        weights = eqx.error_if(weights, jnp.any(jnp.isnan(weights)), "weights are nans")
        assert weights.size == len(hypers)
        out = (embeddings * weights[:, None]).sum(axis=0)
        out = eqx.error_if(out, jnp.any(jnp.isnan(out)), "out of combination of embeddings is nans")
        assert out.size == embeddings[0].size
        return out

    def eval(self, curve_hypers, curves, curve_cutoffs, target_hyper, target_x, target_y):
        curve_embeds_raw = self.generate_embeddings(curves)

        curve_embeds_raw = eqx.error_if(
            curve_embeds_raw, jnp.any(jnp.isnan(curve_embeds_raw)), "nans in embeds"
        )
        curve_embeds = eqx.filter_vmap(
            lambda embed_single_curve: eqx.filter_vmap(
                lambda embed_single: self.encoder.conditioner(embed_single, target_x[0])
            )(embed_single_curve)
        )(curve_embeds_raw)

        curve_embeds = eqx.error_if(
            curve_embeds, jnp.any(jnp.isnan(curve_embeds)), "nans in embeds"
        )

        embeds = curve_embeds[jnp.arange(len(curve_cutoffs)), curve_cutoffs]
        combined = self.combine_embeddings(embeds, curve_hypers, target_hyper)

        weights = self.glue(combined)

        # predict from each combined
        hist = self.decoder(weights)
        pdfs = hist.pdf(target_y[0])
        pdfs = eqx.error_if(pdfs, jnp.any(jnp.isnan(pdfs)), "pdfs are nans")

        ll = jnp.mean(jnp.log(pdfs))

        return ll, hist.mean(), hist.var()

    def loss(self, curve_hypers, curves, target_hyper, target_xs, target_ys, *, key):
        key, subkey = jr.split(key)
        target_ys = target_ys[:, 0]
        assert len(target_ys.shape) == 1
        num_combs = 5
        curve_embeds_raw = self.generate_embeddings(curves)

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

            combinations = eqx.error_if(
                combinations, jnp.any(jnp.isnan(combinations)), "nans after combinations"
            )

            # for each combination: combine it
            combined = eqx.filter_vmap(
                lambda comb: self.combine_embeddings(comb, curve_hypers, target_hyper)
            )(combinations)

            combined = eqx.error_if(
                combined, jnp.any(jnp.isnan(combined)), "nans after combine embeddings"
            )

            weights = eqx.filter_vmap(self.glue)(combined)

            # predict from each combined
            histograms = eqx.filter_vmap(lambda x: self.decoder(x))(weights)
            pdfs = eqx.filter_vmap(lambda hist: hist.pdf(target_y))(histograms)
            pdfs = eqx.error_if(pdfs, jnp.any(jnp.isnan(pdfs)), "pdfs are nans")
            assert pdfs.size == num_combs

            return jnp.mean(jnp.log(pdfs))

        out = eqx.filter_vmap(subseq_curves)(
            curve_embeds, target_ys, jr.split(subkey, curve_embeds.shape[0])
        ).mean()

        return out


if __name__ == "__main__":
    T = 50
    key = jr.key(0)
    k1, k2, k3 = jr.split(key, 3)
    sample_hypercube_hp = lambda key: jr.uniform(key, shape=(10,))
    pi_config = PiConfigSet(sample_hypercube_hp, k2)
    masif = MASIF(k1, pi_config=pi_config)

    def train_step(model: MASIF, key):
        k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)
        curve_maker = pi_config.get_config(k4)

        _lambdas = jr.uniform(k1, (n_curves, n_hyps), minval=0.0, maxval=1.0)
        target_lambda = jr.uniform(k3, (n_hyps,), minval=0.0, maxval=1.0)
        _lambdas = (target_lambda + _lambdas) / 2

        subspace_size = jr.randint(k7, (), 1, n_hyps + 1)

        mask = (jnp.arange(n_hyps) < subspace_size).astype(jnp.int32)
        basis = jr.normal(k6, (n_hyps, n_hyps))
        basis = basis / jnp.linalg.norm(basis, axis=1)
        proj_fn = lambda x: jnp.clip((((basis @ x) + 1) / 2) * mask, 1e-6, 1.0 - 1e-5)
        target_lambda = proj_fn(target_lambda)
        _lambdas = jax.vmap(proj_fn)(_lambdas)

        _lambdas = jnp.clip(_lambdas, 0, 1)
        target_lambda = jnp.clip(target_lambda, 0, 1)

        _lambdas = eqx.error_if(_lambdas, jnp.any(jnp.isnan(_lambdas)), "woof")
        target_lambda = eqx.error_if(target_lambda, jnp.any(jnp.isnan(target_lambda)), "meow")

        curves = eqx.filter_vmap(curve_maker.make)(_lambdas, jr.split(k2, len(_lambdas)))
        curves = eqx.error_if(curves, jnp.any(jnp.isnan(curves)), "moo")

        def make_target(key):
            # this function returns the targets: we want generate a few of them
            # to improve training efficiency
            target_x = jr.choice(key, jnp.arange(T))  # uniform sampling of point of interest
            target_y = curve_maker.make(target_lambda, None)[target_x]  # noiseless
            return target_x, target_y

        n_targets = 5
        target_xs, target_ys = jax.vmap(make_target)(jr.split(k5, n_targets))

        target_xs = eqx.error_if(target_xs, jnp.any(jnp.isnan(target_xs)), "kukareku")
        target_ys = eqx.error_if(target_ys, jnp.any(jnp.isnan(target_ys)), "kukareku but for ys")

        losses = model.loss(_lambdas, curves, target_lambda, target_xs, target_ys, key=k4)
        return -losses.mean()

    @eqx.filter_jit
    def train_loss(model, key, batch_size=200):
        return eqx.filter_vmap(lambda k: train_step(model, k))(jr.split(key, batch_size)).mean()

    def find_nan_leaves(pytree):
        # Flatten the pytree into a list of leaves.
        leaves_with_paths = jax.tree_util.tree_leaves_with_path(pytree)
        # Keep only those leaves that are JAX arrays and contain any NaNs.
        nan_leaves = [
            str(leaf_with_path)
            for leaf_with_path in leaves_with_paths
            if eqx.is_inexact_array(leaf_with_path[1]) and jnp.isnan(leaf_with_path[1]).any()
        ]
        return nan_leaves

    def step(model, opt_state, key):
        loss, grads = eqx.filter_value_and_grad(train_loss)(model, key)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        old_bounds = model.decoder.bounds
        model = eqx.apply_updates(model, updates)
        # the following is a dirty trick that i need to fix later... FIXME
        model = eqx.tree_at(lambda m: m.decoder.bounds, model, old_bounds)
        return model, opt_state, loss

    num_steps = 10_000
    schedule = optax.cosine_decay_schedule(
        init_value=5e-4,
        decay_steps=num_steps,
        alpha=0.1,
    )
    optim = optax.apply_if_finite(
        optax.chain(optax.adamw(learning_rate=schedule, weight_decay=1e-5), optax.clip(1.0)),
        max_consequtive_errors=5,
    )
    opt_state = optim.init(eqx.filter(masif, eqx.is_inexact_array))

    wandb.init(project="masif2")
    for i in tqdm(range(num_steps)):
        masif, opt_state, loss = eqx.filter_jit(step)(masif, opt_state, jr.key(i))
        wandb.log({"loss": loss})
        if i % 10 == 9:
            eqx.tree_serialise_leaves(f"masif_{masif.cmethod}.eqx", masif)

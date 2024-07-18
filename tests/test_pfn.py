import equinox as eqx
import jax
import optax
import pytest
from jax import numpy as jnp
from jax import random as jr
from jax import scipy
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from scipy.stats import linregress
from tqdm import tqdm

from masif2 import Prior
from masif2.pfn import PFN, HistogramDecoder, JointEncoder
from masif2.utils import MASIFError

jax.config.update("jax_compilation_cache_dir", ".jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# Tests for the PFN implementation: checking that the distributions are correct,
# no nans are produced, etc etc

# The PFN is a pretty general architecture overall, but "usually" it is
# a (multi-layer) transformer followed by a histogram-style decoder.

# testing transformers is weird, so we will just test the histogram-ish decoder
# the idea is that we want to approximate the prior with the bins: meaning
# if the networks output in terms of logits is uniform, the histogram-decoder
# produces a distribution that is similar to the prior distribution.

# So, the histogram-decoder requires a Prior as input, and then it should construct
# a thing, that would map uniform distribution to the prior distribution


@pytest.mark.parametrize("n_bins", [-1, 0, 1])
def test_histogram_decoder_fails_for_bad_bin_n(n_bins):
    # single bin shall fail, since we avoid the additional complexity of handling
    # this boundary case; minimum num of bins allowed is 2
    with pytest.raises(MASIFError) as _:
        HistogramDecoder(n_bins=n_bins)


def ground_truth_bounds(num_bins):
    prob_per_bin = 1.0 / num_bins
    return scipy.stats.norm.ppf(
        jnp.array([i * prob_per_bin for i in range(num_bins + 1)]),
    )


@pytest.mark.parametrize("n_bins_pair", [(2, 3), (5, 20), (30, 100)])
def test_histogram_decoder_divergence_comparison(priors, n_bins_pair):
    # probabilitic test: can fail on the correct implementation
    # for the gaussian prior 2 bins are very good, then 3 bins is bad,
    # then with small increase it gets worse, and then better again
    # TODO: plot a graph
    # and then there is a wall of "poor tail representation"
    # which is a pretty interesting effect, that was not talked about in the original
    # PFN paper
    # THERE IS AN OPTIMAL SIZE OF THE HISTOGRAM = f(num_samples, prior_distr)

    diffs = []
    for n_bins in n_bins_pair:
        hd = HistogramDecoder(n_bins=n_bins)
        prior = Prior(prior_fn=priors.flat_gaussian)
        samples = prior.sample(
            xs=jnp.arange(1000).astype(jnp.float32),
            n=1000,
            key=jr.PRNGKey(0),
        ).ravel()
        hd = hd.fit(samples)
        # 2 bins -> should model white noise well, since its the same parametric form
        # hd trains 2 halves of the gaussian, the std should be same..
        uniform_logits = jnp.ones((n_bins,))
        histogram = hd(uniform_logits)

        gt_bounds = ground_truth_bounds(n_bins)

        # this is a pretty important check: that the boundaries are "correct"
        # technically this is almost all we need to ensure corretness
        assert jnp.abs(histogram.bounds[1:-2] - gt_bounds[1:-2]).sum() / n_bins < 0.01

        diff = 0
        mean_computed = 0
        n_xs = 1000
        for x in jr.normal(jr.PRNGKey(1), (n_xs,)):
            likelihood_target = scipy.stats.norm.pdf(x, scale=1)
            likelihood_computed = histogram.pdf(x)
            mean_computed += likelihood_computed * x / n_xs
            diff += likelihood_target * jnp.log(
                likelihood_target / (likelihood_computed + 1e-6),
            )
        diff /= n_xs
        diffs.append(diff)
        assert diff < 0.01  # avg 0.05 likelihood diff is a lot
        # diff is a measure of a divergence between the distributions, it should be smol
        # how small? thats why we need calibration_diff: it computes diff with different
        # normal distr, and is a good benchmark for a "good match"
        #
        # this benchmark can fail with a low probability tho

        # another important point: the distribution in general is hard to model,
        # since the true tails are significantly different from the histogram tails

        # but mean should be close to 0, even with bad tails: distrs are symmetric
        assert jnp.abs(mean_computed) < 0.01
    if n_bins_pair[0] == 2:
        assert diffs[0] < diffs[1]  # bins=2 can learn the distr perfectly
    else:
        assert diffs[0] > diffs[1]  # otherwise we assume general perf improv


@pytest.mark.slow()
def test_pfn_linear_regression():  # noqa
    # train pfn to learn linear regression problems
    # meaning, we have a straight line, fixed intercept, that is perturbed
    # by some white noise. we are trying to predict a value at a particular
    # position, not necessarily masked, thats it
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

    test_samples = make_targets(
        samples[:5000],
        jr.split(key_test, 5000),
    )

    batch_size = 1000

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
        ).sum()

    optim = optax.adam(5e-4)
    opt_state = optim.init(eqx.filter(pfn, eqx.is_array))
    # train for a while
    for sample_idx in (pbar := tqdm(range(1000))):
        samples = prior.sample(
            key=jr.PRNGKey(sample_idx),
            xs=g_xs,
            n=batch_size,
        )
        _sample = make_targets(
            samples,
            jr.split(jr.PRNGKey(1000000 - sample_idx), batch_size),
        )

        loss, grads = eqx.filter_value_and_grad(eqx.Partial(nll, sample=_sample))(
            pfn,
        )
        loss /= batch_size
        pbar.set_description(f"loss : {loss}")
        updates, opt_state = optim.update(grads, opt_state, pfn)
        pfn = eqx.apply_updates(pfn, updates)

        assert not jnp.any(jnp.isnan(pfn.encoder.value_embedder.layer.weight))

    # evalling
    eval_nll = 0
    for sample_idx in tqdm(range(len(test_samples.xs) // 100)):
        _sample = jax.tree.map(
            lambda x: jnp.array(x[sample_idx * 100 : sample_idx * 100 + 100]),
            test_samples,
        )
        eval_nll += nll(pfn, _sample)
    eval_nll /= len(test_samples.xs)

    # compute perfect per-sample nll
    perfect_nll = 0
    for i in tqdm(range(len(test_samples.xs))):
        sample = jax.tree.map(lambda x: x[i], test_samples)
        xs = sample.xs
        ys = sample.ys
        mask = sample.mask
        target_x = sample.target_x
        _target_y = sample._target_y  # noqa: SLF001
        reg = linregress(xs[mask], ys[mask])
        pred = reg.intercept + reg.slope * target_x  # type: ignore
        likelihood = scipy.stats.norm.pdf(_target_y, loc=pred, scale=1)
        perfect_nll += -jnp.log(likelihood)
    perfect_nll /= len(test_samples.xs)

    # a wonderful question is what is a good nll;
    # we have a upper bound (perfect perf), but not really a lower bound
    # we will consider trained nll to be worse than perfect at most by a factor of two
    # which is ln2 of nll, which rounds up to 0.7
    assert eval_nll < perfect_nll + 0.7

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

import pytest
from jax import numpy as jnp
from jax import random as jr
from jax import scipy

from masif2 import Prior
from masif2.pfn import HistogramDecoder
from masif2.utils import MASIFError


@pytest.mark.parametrize("n_bins", [-1, 0, 1])
def test_histogram_decoder_fails_for_bad_bin_n(n_bins):
    # single bin shall fail, since we avoid the additional complexity of handling
    # this boundary case; minimum num of bins allowed is 2
    with pytest.raises(MASIFError) as _:
        HistogramDecoder(n_bins=n_bins)


def ground_truth_bounds(num_bins: int):
    prob_per_bin = 1.0 / num_bins

    return scipy.stats.norm.ppf(
        jnp.array([i * prob_per_bin for i in range(num_bins + 1)]),
    )


@pytest.mark.parametrize("n_bins_pair", [(2, 3), (10, 100)])
def test_histogram_decoder_two_bin(priors, n_bins_pair):
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

        assert jnp.abs(histogram.bounds[1:-2] - gt_bounds[1:-2]).sum() / n_bins < 0.001

        diff = 0
        mean_computed = 0
        n_xs = 1000
        for x in jr.normal(jr.PRNGKey(1), (n_xs,)):
            likelihood_target = scipy.stats.norm.pdf(x, scale=1)
            likelihood_computed = histogram.pdf(x)
            mean_computed += likelihood_computed * x / n_xs
            diff += jnp.abs(likelihood_target - likelihood_computed)
        diff /= n_xs
        diffs.append(diff)
        # diff is a measure of a divergence between the distributions, it should be small.
        # how small? thats why we need calibration_diff: it computes diff with different
        # normal distr, and is a good benchmark for a "good match"
        #
        # this benchmark can fail with a low probability tho

        # another important point: the distribution in general is hard to model,
        # since the true tails are significantly different from the histogram tails

        # but mean should be close to 0 anyway, even with bad tails: distrs are symmetric
        assert jnp.abs(mean_computed) < 0.01
    if n_bins_pair[0] == 2:
        assert diffs[0] < diffs[1]  # smaller n -> larger divergence
    else:
        assert diffs[0] > diffs[1]

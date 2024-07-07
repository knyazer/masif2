# Prior testing: make sure that "smart" sampling gives the same distribution
# as expected, check edge cases, fuzz a little
import pytest
from jax import numpy as jnp
from jax import random as jr

from masif2 import MASIFWarning, Prior

_xs = jnp.arange(100).astype(jnp.float32)


def invalid_prior_posargs(key, length):  # noqa
    return jnp.zeros(length)


def invalid_prior_kwnames(*, keys, length):  # noqa
    return jnp.zeros(length)


@pytest.mark.parametrize("n", [1, 2, 10, 99, 1000])
def test_prior_sampling_with_different_sizes(n, priors):
    prior = Prior(prior_fn=priors.zero)
    assert prior.sample(jr.PRNGKey(0), n=n, xs=_xs).shape[0] == n


@pytest.mark.parametrize("n", [-1, 0, 0.1])
def test_prior_sampling_with_invalid_sizes(n, priors):
    prior = Prior(prior_fn=priors.zero)
    with pytest.raises(Exception):
        prior.sample(jr.PRNGKey(0), n=n, xs=_xs)


@pytest.mark.parametrize("prior_fn", [invalid_prior_posargs, invalid_prior_kwnames])
def test_prior_sampling_with_invalid_args(prior_fn):
    with pytest.raises(Exception):
        Prior(prior_fn=prior_fn).sample(jr.PRNGKey(0), n=100, xs=_xs)


def test_prior_sampling_works_zero(priors):
    prior = Prior(prior_fn=priors.zero)
    assert prior.sample(jr.PRNGKey(0), n=100, xs=_xs).sum() == 0


def test_prior_sampling_works_inc(priors):
    prior = Prior(prior_fn=priors.increasing)
    samples = prior.sample(jr.PRNGKey(0), n=100, xs=_xs)
    assert (samples[:, 0] == 0).all()
    assert (samples[:, -1] == 1).all()


def test_reject_fn_rejects_all(priors):
    def reject_fn(sample, **_):
        return (sample > 0.5).any()  # noqa

    with pytest.warns(MASIFWarning):
        prior = Prior(prior_fn=priors.increasing, reject_fn=reject_fn)
        assert jnp.isnan(prior.sample(jr.PRNGKey(0), n=10, xs=_xs)).all()


def test_reject_fn_rejects_none(priors):
    def reject_fn(sample, **_):
        return (sample > 0.5).any()  # noqa

    prior = Prior(prior_fn=priors.zero, reject_fn=reject_fn)
    assert not jnp.isnan(prior.sample(jr.PRNGKey(0), n=10, xs=_xs)).any()

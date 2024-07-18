import pytest
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, PRNGKeyArray


class TestPriors:
    @staticmethod
    def zero(*, key: PRNGKeyArray, xs: Float[Array, "dim"], **kws):  # noqa
        return jnp.zeros_like(xs)

    @staticmethod
    def increasing(*, key: PRNGKeyArray, xs: Float[Array, "dim"], **kws):  # noqa
        return jnp.linspace(0, 1, xs.shape[0])

    @staticmethod
    def flat_gaussian(*, key: PRNGKeyArray, xs: Float[Array, "dim"], **_):
        # white noise
        return jr.normal(key, (xs.shape[0],))

    @staticmethod
    def increasing_gaussian(*, key: PRNGKeyArray, xs: Float[Array, "dim"], **_):
        return jr.normal(key, (xs.shape[0],)) + jnp.linspace(-10, 10, xs.shape[0])


@pytest.fixture(scope="module")
def priors():
    return TestPriors

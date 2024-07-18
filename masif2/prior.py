from typing import Any

import equinox as eqx
from jax import lax
from jax import numpy as jnp
from jax import random as jr

from masif2.utils import warn


class Prior(eqx.Module):
    prior_fn: Any  # anything that has a __call__ method
    reject_fn: Any

    def __init__(self, /, prior_fn=None, reject_fn=None):
        assert prior_fn is not None
        self.prior_fn = prior_fn
        if reject_fn is None:
            self.reject_fn = lambda **_: False  # reject nothing
        else:
            self.reject_fn = reject_fn

    def prior_or_nan(self, /, **kws):
        # return either a sample from the prior_fn, or None if reject_fn decides so
        sample = self.prior_fn(**kws)
        return lax.cond(
            self.reject_fn(sample=sample, **kws),
            lambda: sample * jnp.nan,
            lambda: sample,
        )

    @eqx.filter_jit
    def sample(self, /, key=None, n: int = 0, xs=None, redundancy=4):
        assert key is not None
        assert n >= 1
        assert isinstance(redundancy, int)
        assert redundancy >= 1, "redundancy < 1 does not make sense"

        key_samples, key_choice = jr.split(key, 2)

        @eqx.filter_vmap
        def vmapped_prior_or_nan(key):
            return self.prior_or_nan(key=key, n=n, xs=xs)

        # we don't expose the redundancy offset as an arg
        # since the less one knows, the better. Also, it is not important really
        to_sample = int(n * redundancy + 10)

        samples_with_redundancy = vmapped_prior_or_nan(jr.split(key_samples, to_sample))

        if xs is None:
            rejected_mask = jnp.isnan(samples_with_redundancy)
        else:
            rejected_mask = jnp.any(jnp.isnan(samples_with_redundancy), axis=1)
        accepted_mask = jnp.logical_not(rejected_mask)

        # probability for choosing each of the sequences: zeros for nans
        # uniform over all others otherwise: even if we somehow got less samples than
        # we would like to, the distr will be uniform, tho with repeats but unbiased
        num_valid_samples = jnp.count_nonzero(accepted_mask)
        probs = accepted_mask / num_valid_samples
        # detect if anything went wrong (e.g. division by zero)
        are_probs_valid = jnp.isclose(jnp.sum(probs), 1.0)

        def choice_closure(*, replace):
            # chooses between "sample with replacement" or "sample without replacement"
            # also makes 'replace' arg non-static with a trick
            def _closure(arg):
                return jr.choice(
                    key_choice,
                    samples_with_redundancy,
                    shape=(n,),
                    replace=arg,
                    p=probs,
                    axis=0,
                )

            return lax.cond(
                replace,
                lambda: _closure(arg=True),
                lambda: _closure(arg=False),
            )

        def complain_and_give_up():
            # this function is called only if there were no samples generated
            # usually this means that the situation is very bad, like,
            # this will screw up everything later on, without almost any feedback
            warn(
                """
            not_enough_samples:
                The rejection_fn rejects too much.
                Another (unlikely) cause might be prior_fn returning nans a lot.
                Try to reduce the rejection rate, or increase the redundancy.
                ! This warning can be ignored, but it better not be.
            """.replace("\r", ""),
            )
            if xs is None:
                return jnp.nan * jnp.zeros((n,))
            return jnp.nan * jnp.zeros((n, xs.shape[0]))

        samples = lax.cond(
            are_probs_valid,
            lambda: choice_closure(replace=(num_valid_samples <= n)),
            complain_and_give_up,
        )

        return samples

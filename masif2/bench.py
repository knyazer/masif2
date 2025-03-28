"""
from .evals.ifbo import MASIF, PiConfigSet
from jax import random as jr
import equinox as eqx

sample_hypercube_hp = lambda key: jr.uniform(key, shape=(3,))
pi_config = PiConfigSet(sample_hypercube_hp, jr.key(1))
m = MASIF(jr.key(0), pi_config=pi_config)
m = eqx.tree_deserialise_leaves("masif.eqx", m)

"""

# ruff: noqa
import jax
from beartype.claw import beartype_this_package

beartype_this_package()

# enable fast rng keys, unstable; current jax version: 0.4.31 (check the uv lock file)
# when I am running on TPU I install nightly, so I don't know what exact commit it will be.
jax.config.update("jax_threefry_partitionable", True)  # noqa

# enable compilation cache: faster compiling
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")  # noqa
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# enable strict numpy dtype conversion rules: easy-to-debug mixed precision
# jax.config.update("jax_numpy_dtype_promotion", "strict")

from .decoders import Decoder, HistogramDecoder
from .encoders import Encoder, JointEncoder
from .model import PFN

__all__ = [
    "HistogramDecoder",
    "JointEncoder",
    "Encoder",
    "Decoder",
    "PFN",
]

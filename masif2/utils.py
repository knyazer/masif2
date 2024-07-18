import functools
from warnings import warn as pywarn

import jax


class MASIFWarning(Warning): ...


class MASIFError(Exception): ...


def _pywarn_wrapper(s: str):
    return pywarn(s, MASIFWarning, stacklevel=1)


def warn(s: str):
    jax.debug.callback(functools.partial(_pywarn_wrapper, s))

from beartype.claw import beartype_this_package

from masif2.prior import Prior
from masif2.utils import MASIFWarning

beartype_this_package()


__all__ = ["Prior", "MASIFWarning"]

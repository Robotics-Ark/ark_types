from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# bring the raw message classes into the namespace
# ----------------------------------------------------------------------
from _arktypes_std.flag_t import flag_t as _flag_t
from _arktypes_std.float_array_t import float_array_t as _float_array_t


# ----------------------------------------------------------------------
# helpers specific to flag_t
# ----------------------------------------------------------------------
def _add_flag_helpers(cls):

    # -------- read --------
    def __int__(self):
        """Convert message to integer."""
        return int(self.data)

    # -------- factory --------
    @classmethod
    def from_int(c, d: int):
        """Create message from integer."""
        obj = c()
        obj.data = int(d)
        return obj

    cls.__int__ = __int__
    cls.from_int = from_int

    return cls


# ----------------------------------------------------------------------
# helpers specific to float_array_t
# ----------------------------------------------------------------------
def _add_float_array_helpers(cls):

    # -------- read --------
    def as_array(self):
        """Convert to numpy array."""
        return np.array(self.data)

    # -------- factory --------
    @classmethod
    def from_array(c, array: np.ndarray | list[float]):
        """Initialize from a numpy array or list of floats."""

        obj = c()

        array = np.asarray(array).flatten()
        obj.n = int(array.shape[0])
        obj.data = array.tolist()

        return obj

    cls.as_array = as_array
    cls.from_array = from_array

    return cls


# ----------------------------------------------------------------------
# patch the raw classes in‑place
# ----------------------------------------------------------------------
_add_flag_helpers(_flag_t)
_add_float_array_helpers(_float_array_t)

# ----------------------------------------------------------------------
# re‑export under user‑friendly names
# ----------------------------------------------------------------------
flag_t = _flag_t
float_array_t = _float_array_t

__all__ = [
    "flag_t",
    "float_array_t",
]

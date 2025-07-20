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
    """
    Augments a `flag_t` class with helper methods for reading and constructing objects.

    Adds:
        - `__int__`: Casts the `flag_t` instance to an integer.
        - `from_int`: Class method to construct a `flag_t` instance from an integer.

    Args:
        cls: The class object to augment (expected to be `_flag_t`).

    Returns:
        The class with added helper methods.
    """

    def __int__(self):
        """
        Convert the `flag_t` instance to an integer by returning the value of the `data` attribute.

        Returns:
            int: Integer representation of the flag.
        """
        return int(self.data)

    @classmethod
    def from_int(c, d: int):
        """
        Create a `flag_t` instance from an integer.

        Args:
            d (int): Integer value to initialize the flag.

        Returns:
            flag_t: A new instance with the `data` field set to the given integer.
        """
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
    """
    Augments a `float_array_t` class with helper methods for array conversion and construction.

    Adds:
        - `as_array`: Converts the instance's data to a NumPy array.
        - `from_array`: Class method to construct a `float_array_t` instance from a NumPy array or list.

    Args:
        cls: The class object to augment (expected to be `_float_array_t`).

    Returns:
        The class with added helper methods.
    """

    def as_array(self):
        """
        Convert the `float_array_t` instance's internal list to a NumPy array.

        Returns:
            np.ndarray: A 1D NumPy array of floats.
        """
        return np.array(self.data, dtype=float)

    @classmethod
    def from_array(c, array: np.ndarray | list[float]):
        """
        Create a `float_array_t` instance from a 1D NumPy array or list of floats.

        Args:
            array (np.ndarray | list[float]): The input array or list of float values.

        Returns:
            float_array_t: A new instance with `n` and `data` fields set accordingly.

        Raises:
            AssertionError: If the input array is not one-dimensional.
        """
        obj = c()

        array = np.asarray(array, dtype=float)
        assert array.ndim == 1, "Input must be a 1D array or list"
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

__all__ = ["flag_t", "float_array_t"]

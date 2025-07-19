import numpy as np
import pytest

from arktypes import float_array_t, flag_t


# -----------------------------------------------------------------------------
# float_array_t helpers
# -----------------------------------------------------------------------------


def test_float_array_basic_properties():
    """from_array should populate n, data, and dtype correctly."""
    arr = float_array_t.from_array([1, 2, 3])

    # structural checks
    assert arr.n == 3
    assert arr.data == [1.0, 2.0, 3.0]
    assert all(isinstance(x, float) for x in arr.data)

    # numpy view
    np.testing.assert_array_equal(arr.as_array(), np.array([1, 2, 3], dtype=float))
    # helper should always promote to float64 for precision
    assert arr.as_array().dtype == np.float64


def test_float_array_encode_decode_roundtrip():
    """Binary encode / decode should be loss‑less."""
    original = float_array_t.from_array([1.5, 2.5, 3.5])

    encoded = original.encode()
    decoded = float_array_t.decode(encoded)

    # decoded.data might be a tuple depending on LCM version – convert both to list
    assert list(decoded.data) == original.data
    assert decoded.n == original.n
    np.testing.assert_array_equal(decoded.as_array(), original.as_array())


def test_float_array_from_ndarray_input():
    """Factory must accept numpy arrays of any dtype and cast to float."""
    np_arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    obj = float_array_t.from_array(np_arr)

    assert obj.n == 3
    # as_array() should match after promotion to float64
    np.testing.assert_array_equal(obj.as_array(), np_arr.astype(float))


def test_float_array_invalid_dimension_raises():
    """Passing a 2‑D array should raise an AssertionError (see helper implementation)."""
    with pytest.raises(AssertionError):
        float_array_t.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))


# -----------------------------------------------------------------------------
# flag_t helpers
# -----------------------------------------------------------------------------


def test_flag_int_conversion_and_roundtrip():
    """from_int and __int__ should be inverses and survive encode/decode."""
    val = 123456789
    obj = flag_t.from_int(val)

    assert int(obj) == val

    encoded = obj.encode()
    decoded = flag_t.decode(encoded)
    assert int(decoded) == val


@pytest.mark.parametrize("value", [-1, 0, 42, 2**63 - 1, -(2**63)])
def test_flag_handles_various_int_ranges(value):
    """Helper should work for a representative range of 64‑bit signed ints."""
    obj = flag_t.from_int(value)
    assert int(obj) == value

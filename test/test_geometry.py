import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.spatial.transform import Rotation as Rot

from arktypes import (
    vector3_t,
    quaternion_t,
    transform_t,
    twist_t,
    wrench_t,
)


# -----------------------------------------------------------------------------
# vector3_t
# -----------------------------------------------------------------------------


def test_vector3_from_array_and_roundtrip():
    v = vector3_t.from_array([1, 2, 3])

    assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0
    np.testing.assert_array_equal(v.as_array(), [1.0, 2.0, 3.0])

    encoded = v.encode()
    decoded = vector3_t.decode(encoded)
    np.testing.assert_array_equal(decoded.as_array(), v.as_array())


def test_vector3_identity():
    v = vector3_t.identity()
    np.testing.assert_array_equal(v.as_array(), [0.0, 0.0, 0.0])


def test_vector3_from_array_wrong_length_raises():
    with pytest.raises(ValueError):
        vector3_t.from_array([1, 2])


# -----------------------------------------------------------------------------
# quaternion_t
# -----------------------------------------------------------------------------


def _quat_equal(q1, q2, tol=1e-8):
    """Helper to compare quaternions accounting for sign ambiguity."""
    q1, q2 = np.asarray(q1), np.asarray(q2)
    return np.allclose(q1, q2, atol=tol) or np.allclose(q1, -q2, atol=tol)


def test_quaternion_from_rotation_and_roundtrip():
    rot = Rot.from_euler("xyz", [30, 45, 60], degrees=True)
    q = quaternion_t.from_rotation(rot)

    # Check forward conversion
    assert _quat_equal(q.as_rotation().as_quat(), rot.as_quat())

    # binary round‑trip
    encoded = q.encode()
    decoded = quaternion_t.decode(encoded)
    assert _quat_equal(decoded.as_rotation().as_quat(), rot.as_quat())


def test_quaternion_identity():
    q = quaternion_t.identity()
    assert _quat_equal(q.as_rotation().as_quat(), [0, 0, 0, 1])


def test_quaternion_from_rotation_matrix():
    R = Rot.from_euler("y", 90, degrees=True).as_matrix()
    q = quaternion_t.from_rotation(R)
    # Verify conversion gives same matrix back
    np.testing.assert_allclose(q.as_rotation().as_matrix(), R, atol=1e-8)


def test_quaternion_invalid_shape_raises():
    with pytest.raises(ValueError):
        quaternion_t.from_rotation(np.eye(4))  # wrong shape


def test_quaternion_invalid_type_raises():
    with pytest.raises(TypeError):
        quaternion_t.from_rotation("not a rotation")


# -----------------------------------------------------------------------------
# transform_t
# -----------------------------------------------------------------------------


def test_transform_identity_and_roundtrip():
    tf = transform_t.identity()
    np.testing.assert_array_equal(tf.as_array(), np.eye(4))

    encoded = tf.encode()
    decoded = transform_t.decode(encoded)
    np.testing.assert_array_almost_equal(decoded.as_array(), np.eye(4))


def test_transform_from_array():
    # translation [1,2,3], identity rotation
    mat = np.eye(4)
    mat[:3, 3] = [1.0, 2.0, 3.0]

    tf = transform_t.from_array(mat)

    np.testing.assert_array_equal(tf.translation.as_array(), [1, 2, 3])
    np.testing.assert_array_equal(tf.rotation.as_rotation().as_matrix(), np.eye(3))
    np.testing.assert_array_equal(tf.as_array(), mat)


# -----------------------------------------------------------------------------
# twist_t
# -----------------------------------------------------------------------------


def test_twist_from_array_and_as_array():
    data = [1, 2, 3, 0.1, 0.2, 0.3]

    tw = twist_t.from_array(data)  # linear first
    np.testing.assert_array_almost_equal(tw.as_array(), data)

    tw2 = twist_t.from_array(data, linear_first=False)
    # With reverse ordering the first 3 elements are angular
    np.testing.assert_array_almost_equal(tw2.as_array(linear_first=False), data)

    # round‑trip encode/decode
    enc = tw.encode()
    dec = twist_t.decode(enc)
    np.testing.assert_array_almost_equal(dec.as_array(), data)


@pytest.mark.parametrize("bad_len", [5, 7])
def test_twist_invalid_length_raises(bad_len):
    with pytest.raises(ValueError):
        twist_t.from_array(np.arange(bad_len))


# -----------------------------------------------------------------------------
# wrench_t
# -----------------------------------------------------------------------------


def test_wrench_from_array_and_as_array():
    data = [10, 20, 30, 0.4, 0.5, 0.6]

    wh = wrench_t.from_array(data)  # force first
    np.testing.assert_array_almost_equal(wh.as_array(), data)

    wh2 = wrench_t.from_array(data, force_first=False)
    np.testing.assert_array_almost_equal(wh2.as_array(force_first=False), data)

    # encode/decode
    enc = wh.encode()
    dec = wrench_t.decode(enc)
    np.testing.assert_array_almost_equal(dec.as_array(), data)


@pytest.mark.parametrize("bad_len", [0, 4, 8])
def test_wrench_invalid_length_raises(bad_len):
    with pytest.raises(ValueError):
        wrench_t.from_array(np.arange(bad_len))

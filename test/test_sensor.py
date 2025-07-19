import numpy as np
import pytest
from pathlib import Path

# Pillow (PIL) is an optional dependency – abort sensor tests if missing
PIL = pytest.importorskip("PIL")
from PIL import Image

from arktypes import joint_state_t, image_t, laser_scan_t
from arktypes._sensor import joy_t  # joy_t not re‑exported at package root


# -----------------------------------------------------------------------------
# joint_state_t
# -----------------------------------------------------------------------------


def test_joint_state_init_and_roundtrip():
    names = ["hip", "knee", "ankle"]
    pos = [0.1, 0.2, 0.3]
    vel = [0.0, 0.0, 0.0]
    eff = [1.0, 1.0, 1.0]
    ext = [0.05, 0.05, 0.05]

    js = joint_state_t.init(names, pos, vel, eff, ext)

    # structural checks -------------------------------------------------------
    assert js.ndof == 3
    assert js.name == names
    assert js.position == pos
    assert js.velocity == vel
    assert js.effort == eff
    assert js.external_torque == ext

    # helper getter (index‑based; name‑based path currently buggy upstream) ---
    assert js.get_position(1) == pos[1]
    assert js.get_velocity(2) == vel[2]

    # binary round‑trip -------------------------------------------------------
    enc = js.encode()
    dec = joint_state_t.decode(enc)
    assert dec.ndof == js.ndof
    assert np.allclose(list(dec.position), pos)
    assert np.allclose(list(dec.velocity), vel)


# -----------------------------------------------------------------------------
# joy_t
# -----------------------------------------------------------------------------


def test_joy_axis_index_access_and_roundtrip():
    j = joy_t.init(
        axis_names=["x", "y"], axes=[-1.0, 0.5], button_names=["A"], buttons=[1]
    )

    # index‑based accessor ----------------------------------------------------
    assert pytest.approx(j.get_axis(0)) == -1.0
    assert pytest.approx(j.get_axis(1)) == 0.5

    enc = j.encode()
    dec = joy_t.decode(enc)
    assert dec.naxes == j.naxes
    assert pytest.approx(dec.axes[1]) == 0.5


# -----------------------------------------------------------------------------
# image_t  – ndarray path
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "array, expected_mode",
    [
        (np.random.randint(0, 256, (4, 4), dtype=np.uint8), "L"),
        (np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8), "RGB"),
    ],
)
def test_image_from_array_roundtrip(array, expected_mode):
    msg = image_t.from_array(array, frame_name="cam")

    # basic fields -----------------------------------------------------------
    assert msg.width == array.shape[1]
    assert msg.height == array.shape[0]
    np.testing.assert_array_equal(msg.as_array(), array)

    # PIL conversion ---------------------------------------------------------
    pil_img = msg.as_image()
    assert pil_img.mode == expected_mode
    np.testing.assert_array_equal(
        np.asarray(pil_img), array if array.ndim == 3 else array
    )

    # binary round‑trip ------------------------------------------------------
    enc = msg.encode()
    dec = image_t.decode(enc)
    np.testing.assert_array_equal(dec.as_array(copy=True), array)


def test_image_as_array_with_compression_not_supported():
    arr = np.zeros((2, 2), dtype=np.uint8)
    msg = image_t.from_array(arr)
    msg.compression_method = image_t.COMPRESSION_METHOD_ZLIB
    with pytest.raises(NotImplementedError):
        msg.as_array()


def test_image_from_array_mismatched_pixel_format_raises():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        image_t.from_array(
            rgb, pixel_format=image_t.PIXEL_FORMAT_RGBA
        )  # 4‑channel format vs 3‑channel data


# -----------------------------------------------------------------------------
# image_t – real file path (kuka.png)
# -----------------------------------------------------------------------------


def test_image_from_image_file_roundtrip():
    """Verify from_image/as_image preserve data using the provided test image."""
    path = Path(__file__).with_name("kuka.png")
    img = Image.open(path)

    msg = image_t.from_image(img, frame_name="camera")

    arr = np.asarray(img)
    np.testing.assert_array_equal(msg.as_array(), arr)

    # round‑trip back to PIL --------------------------------------------------
    img2 = msg.as_image()
    np.testing.assert_array_equal(np.asarray(img2), arr)

    # encoded->decoded consistency ------------------------------------------
    enc = msg.encode()
    dec = image_t.decode(enc)
    np.testing.assert_array_equal(dec.as_array(copy=True), arr)


# -----------------------------------------------------------------------------
# laser_scan_t
# -----------------------------------------------------------------------------


def test_laser_scan_init_and_roundtrip():
    angles = np.linspace(-1.0, 1.0, 5)
    ranges = np.linspace(0.5, 2.5, 5)
    scan = laser_scan_t.init(angles, ranges)

    assert scan.n == 5
    np.testing.assert_array_almost_equal(scan.angles, angles)
    np.testing.assert_array_almost_equal(scan.ranges, ranges)

    # helpers provided by wrapper -------------------------------------------
    np.testing.assert_array_almost_equal(scan.angles.as_array(), angles)
    np.testing.assert_array_almost_equal(scan.ranges.as_array(), ranges)

    # binary round‑trip ------------------------------------------------------
    enc = scan.encode()
    dec = laser_scan_t.decode(enc)

    # decoded lists are wrapped in custom ArrayList with .as_array() ---------
    assert hasattr(dec.angles, "as_array")
    np.testing.assert_array_almost_equal(dec.angles.as_array(), angles)
    np.testing.assert_array_almost_equal(dec.ranges.as_array(), ranges)


def test_laser_scan_mismatched_lengths_raise_value_error():
    with pytest.raises(ValueError):
        laser_scan_t.init([0.0, 1.0], [1.0])

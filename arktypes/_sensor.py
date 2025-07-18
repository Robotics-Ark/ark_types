from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

# ----------------------------------------------------------------------
# bring the raw message classes into the namespace
# ----------------------------------------------------------------------
from _arktypes_sensor.joint_state_t import joint_state_t as _joint_state_t
from _arktypes_sensor.joy_t import joy_t as _joy_t
from _arktypes_sensor.image_t import image_t as _image_t
from _arktypes_sensor.laser_scan_t import laser_scan_t as _laser_scan_t
from _arktypes_sensor.imu_t import imu_t as _imu_t


# ----------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------
def _get_named_item(
    self,
    name_attr: str,
    name_attr_array: str,
    x: str | int | list[str] | list[int],
):
    if isinstance(x, int):
        arr = getattr(self, name_attr_array)
        return arr[x]
    elif isinstance(x, str):
        name_arr = getattr(self, name_attr)
        return _get_named_item(self, attr_name, name_attr_array, name_arr.index(x))
    elif isinstance(x, list):
        return [_get_named_item(self, name_attr, name_attr_array) for i in x]
    else:
        raise ValueError(f"x must be a 'str' or 'int', got {type(x)}")


# ----------------------------------------------------------------------
# helpers specific to joint_state_t
# ----------------------------------------------------------------------
def _add_joint_state_helpers(cls):

    # -------- read --------
    def get_position(self, p: str | int):
        return _get_named_item(self, "name", "position", p)

    def get_velocity(self, v: str | int):
        return _get_named_item(self, "name", "velocity", v)

    def get_effort(self, e: str | int):
        return _get_named_item(self, "name", "effort", e)

    def get_external_torque(self, e: str | int):
        return _get_named_item(self, "name", "external_torque", e)

    # -------- factory --------
    @classmethod
    def init(
        c,
        name: list[str],
        position: list[float] | np.ndarray,
        velocity: list[float] | np.ndarray = None,
        effort: list[float] | np.ndarray = None,
        external_torque: list[float] | np.ndarray = None,
    ):

        ndof = len(name)

        def parse(x):
            if x is not None:
                return np.asarray(x).flatten().tolist()
            else:
                return np.zeros(ndof)

        obj = c()
        obj.ndof = ndof
        obj.name = name
        obj.position = parse(position)
        obj.velocity = parse(velocity)
        obj.effort = parse(effort)
        obj.external_torque = parse(external_torque)

        return obj

    cls.get_position = get_position
    cls.get_velocity = get_velocity
    cls.get_effort = get_effort
    cls.get_external_torque = get_external_torque
    cls.init = init

    return cls


# ----------------------------------------------------------------------
# helpers specific to joy_t
# ----------------------------------------------------------------------
def _add_joy_helpers(cls):

    # -------- read --------
    def get_axis(self, a: str | int | list[str] | list[int]):
        return _get_named_item(self, "axis_names", "axes", a)

    def get_button(self, b: str | int | list[str] | list[int]):
        return _get_named_item(self, "button_names", "buttons", a)

    # -------- factory --------
    @classmethod
    def init(
        c,
        axis_names: list[str],
        axes: list[float] = [],
        button_names: list[str] = [],
        buttons: list[int] = [],
    ):
        obj = c()
        obj.naxes = len(axes)
        obj.axes = axes
        obj.axis_names = axis_names
        obj.nbuttons = len(buttons)
        obj.buttons = buttons
        obj.button_names = button_names
        return obj

    cls.init = init
    cls.get_axis = get_axis
    cls.get_button = get_button

    return cls


# ----------------------------------------------------------------------
# helpers specific to image_t
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# helpers specific to image_t
# ----------------------------------------------------------------------
def _add_image_helpers(cls):
    """
    Patch the raw `image_t` class in-place with:
        * as_array()  → numpy.ndarray            (no copy by default)
        * as_pil()    → PIL.Image                (RGB/RGBA/L as appropriate)
        * from_array() → `image_t` factory       (uncompressed)
        * from_pil()   → `image_t` factory       (uncompressed)

    Only the “not-compressed” path is handled; any other compression_method
    raises NotImplementedError.
    """

    # ------------------------------------------------------------------
    # internal lookup tables
    # ------------------------------------------------------------------
    _channel_type_to_dtype = {
        cls.CHANNEL_TYPE_INT8: np.int8,
        cls.CHANNEL_TYPE_UINT8: np.uint8,
        cls.CHANNEL_TYPE_INT16: np.int16,
        cls.CHANNEL_TYPE_UINT16: np.uint16,
        cls.CHANNEL_TYPE_INT32: np.int32,
        cls.CHANNEL_TYPE_UINT32: np.uint32,
        cls.CHANNEL_TYPE_FLOAT32: np.float32,
        cls.CHANNEL_TYPE_FLOAT64: np.float64,
    }
    _dtype_to_channel_type = {v: k for k, v in _channel_type_to_dtype.items()}

    _pixel_format_to_channels = {
        cls.PIXEL_FORMAT_GRAY: 1,
        cls.PIXEL_FORMAT_RGB: 3,
        cls.PIXEL_FORMAT_BGR: 3,
        cls.PIXEL_FORMAT_RGBA: 4,
        cls.PIXEL_FORMAT_BGRA: 4,
        cls.PIXEL_FORMAT_DEPTH: 1,
        cls.PIXEL_FORMAT_LABEL: 1,
        cls.PIXEL_FORMAT_MASK: 1,
        cls.PIXEL_FORMAT_DISPARITY: 1,
        cls.PIXEL_FORMAT_BAYER_BGGR: 1,
        cls.PIXEL_FORMAT_BAYER_RGGB: 1,
        cls.PIXEL_FORMAT_BAYER_GBRG: 1,
        cls.PIXEL_FORMAT_BAYER_GRBG: 1,
    }
    _channels_to_default_pixel_format = {
        1: cls.PIXEL_FORMAT_GRAY,
        3: cls.PIXEL_FORMAT_RGB,
        4: cls.PIXEL_FORMAT_RGBA,
    }

    # ------------------------------------------------------------------
    # helpers – read
    # ------------------------------------------------------------------
    def as_array(self, *, copy: bool = False) -> np.ndarray:
        """Return image data as a NumPy array of shape (H,W[,C])."""
        if self.compression_method != cls.COMPRESSION_METHOD_NOT_COMPRESSED:
            raise NotImplementedError("Only uncompressed images are supported")

        # dtype ----------------------------------------------------------------
        dtype = _channel_type_to_dtype.get(self.channel_type)
        if dtype is None:
            raise ValueError(f"Unsupported channel_type {self.channel_type}")

        # number of channels ----------------------------------------------------
        nch = _pixel_format_to_channels.get(self.pixel_format)
        if nch is None:
            raise ValueError(f"Unsupported pixel_format {self.pixel_format}")

        # view into the raw buffer ---------------------------------------------
        buf = memoryview(self.data)
        flat = np.frombuffer(buf, dtype=dtype)

        # handle row-stride / padding ------------------------------------------
        exp_row_bytes = self.width * nch * dtype().nbytes
        if self.row_stride not in (0, exp_row_bytes):
            # rows contain padding -> build strided view then crop
            total_elems = (self.row_stride // dtype().nbytes) * self.height
            flat = flat[:total_elems]  # safety
            arr = flat.reshape(self.height, self.row_stride // dtype().nbytes)
            arr = arr[:, : self.width * nch]
        else:
            arr = flat

        # reshape to (H,W[,C]) --------------------------------------------------
        if nch == 1:
            arr = arr.reshape(self.height, self.width)
        else:
            arr = arr.reshape(self.height, self.width, nch)

        # endian swap if required ----------------------------------------------
        if self.bigendian and arr.dtype.isnative:
            arr = arr.byteswap(inplace=False)

        return arr.copy() if copy else arr

    def as_pil(self) -> Image.Image:
        """Return a `PIL.Image` view of the data (RGB/RGBA/L)."""
        arr = self.as_array(copy=False)

        pf = self.pixel_format
        if pf in (cls.PIXEL_FORMAT_BGR, cls.PIXEL_FORMAT_BGRA):
            # convert BGR(A) -> RGB(A)
            if arr.ndim != 3:
                raise ValueError("BGR/BGRA data must have 3D shape")
            if pf == cls.PIXEL_FORMAT_BGR:
                arr = arr[..., ::-1]  # B,G,R → R,G,B
                mode = "RGB"
            else:  # BGRA
                arr = arr[..., [2, 1, 0, 3]]  # B,G,R,A → R,G,B,A
                mode = "RGBA"
        elif pf == cls.PIXEL_FORMAT_RGB:
            mode = "RGB"
        elif pf == cls.PIXEL_FORMAT_RGBA:
            mode = "RGBA"
        else:
            # fallback -> treat as single-channel luminance
            mode = "L"
            if arr.ndim == 3:
                arr = arr[..., 0]

        return Image.fromarray(arr, mode=mode)

    # ------------------------------------------------------------------
    # helpers – factories
    # ------------------------------------------------------------------
    @classmethod
    def from_array(
        c, array: np.ndarray, *, frame_name: str = "", pixel_format: int | None = None
    ) -> "image_t":
        """
        Build an `image_t` from a NumPy array.

        `pixel_format` is inferred from array.shape if not provided.
        """
        if array.ndim not in (2, 3):
            raise ValueError("array must have rank 2 or 3")

        # infer pixel format ----------------------------------------------------
        if pixel_format is None:
            pixel_format = _channels_to_default_pixel_format.get(
                1 if array.ndim == 2 else array.shape[2]
            )
            if pixel_format is None:
                raise ValueError("Cannot deduce pixel_format from array shape")

        nch = _pixel_format_to_channels[pixel_format]
        if (array.ndim == 2 and nch != 1) or (
            array.ndim == 3 and array.shape[2] != nch
        ):
            raise ValueError("array shape does not match pixel_format")

        # dtype / channel_type --------------------------------------------------
        channel_type = _dtype_to_channel_type.get(array.dtype.type)
        if channel_type is None:
            raise ValueError(f"Unsupported dtype {array.dtype}")

        # ensure contiguous -----------------------------------------------------
        buf = np.ascontiguousarray(array)
        height, width = buf.shape[:2]
        row_stride = buf.strides[0]

        # populate message ------------------------------------------------------
        msg = c()
        msg.frame_name = frame_name
        msg.width = int(width)
        msg.height = int(height)
        msg.row_stride = int(row_stride)
        msg.size = buf.nbytes
        msg.data = buf.tobytes()
        msg.bigendian = not buf.dtype.isnative
        msg.pixel_format = int(pixel_format)
        msg.channel_type = int(channel_type)
        msg.compression_method = cls.COMPRESSION_METHOD_NOT_COMPRESSED
        return msg

    @classmethod
    def from_pil(c, img: Image.Image, *, frame_name: str = "") -> "image_t":
        """Build an `image_t` from a `PIL.Image`."""
        mode_to_pf = {
            "L": cls.PIXEL_FORMAT_GRAY,
            "RGB": cls.PIXEL_FORMAT_RGB,
            "RGBA": cls.PIXEL_FORMAT_RGBA,
        }
        if img.mode not in mode_to_pf:
            # Convert unsupported modes to a safe default (RGB)
            img = img.convert("RGB")

        pixel_format = mode_to_pf[img.mode]
        arr = np.asarray(img)
        return c.from_array(arr, frame_name=frame_name, pixel_format=pixel_format)

    # ------------------------------------------------------------------
    # patch class -------------------------------------------------------
    cls.as_array = as_array
    cls.as_pil = as_pil
    cls.from_array = from_array
    cls.from_pil = from_pil
    return cls


# ----------------------------------------------------------------------
# helpers specific to laser_scan_helpers_t
# ----------------------------------------------------------------------
def _add_laser_scan_helpers(cls):
    # TODO
    # -------- read --------
    # -------- factory --------
    return cls


# ----------------------------------------------------------------------
# helpers specific to imu_t
# ----------------------------------------------------------------------
def _add_imu_helpers(cls):
    # TODO
    # -------- read --------
    # -------- factory --------
    return cls


# ----------------------------------------------------------------------
# patch the raw classes in-place
# ----------------------------------------------------------------------
_add_joint_state_helpers(_joint_state_t)
_add_joy_helpers(_joy_t)
_add_image_helpers(_image_t)
_add_laser_scan_helpers(_laser_scan_t)
_add_imu_helpers(_imu_t)

# ----------------------------------------------------------------------
# re-export under user-friendly names
# ----------------------------------------------------------------------
joint_state_t = _joint_state_t
joy_t = _joy_t
image_t = _image_t
laser_scan_t = _laser_scan_t
imu_t = _imu_t

__all__ = [
    "joint_state_t",
    "image_t",
    "laser_scan_t",
    "imu_t",
]

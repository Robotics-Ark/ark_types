from __future__ import annotations

import numpy as np
from PIL import Image
from functools import wraps
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
    """
    Retrieve an item from an array attribute using a corresponding name list.

    Args:
        self: The object containing the named fields.
        name_attr: Attribute name holding the list of names.
        name_attr_array: Attribute name holding the values.
        x: A single index, name, or a list of indices/names.

    Returns:
        A single value or a list of values corresponding to the names/indices.

    Raises:
        ValueError: If `x` is not a supported type.
    """
    if isinstance(x, int):
        arr = getattr(self, name_attr_array)
        return arr[x]
    elif isinstance(x, str):
        name_arr = getattr(self, name_attr)
        return _get_named_item(self, name_attr, name_attr_array, name_arr.index(x))
    elif isinstance(x, list):
        return [_get_named_item(self, name_attr, name_attr_array, i) for i in x]
    else:
        raise ValueError(f"x must be a 'str' or 'int', got {type(x)}")


# ----------------------------------------------------------------------
# helpers specific to joint_state_t
# ----------------------------------------------------------------------
def _add_joint_state_helpers(cls):
    """
    Add convenience accessors and initialization helper to `joint_state_t`.

    Adds:
        - get_position, get_velocity, get_effort, get_external_torque
        - init() factory
    """

    # -------- read --------
    def get_position(self, p: str | int):
        """Retrieve position by joint name or index."""
        return _get_named_item(self, "name", "position", p)

    def get_velocity(self, v: str | int):
        """Retrieve velocity by joint name or index."""
        return _get_named_item(self, "name", "velocity", v)

    def get_effort(self, e: str | int):
        """Retrieve effort by joint name or index."""
        return _get_named_item(self, "name", "effort", e)

    def get_external_torque(self, e: str | int):
        """Retrieve external torque by joint name or index."""
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
        """
        Factory for joint_state_t from given name, position, and optional values.

        Args:
            name: List of joint names.
            position: Joint positions.
            velocity: Optional velocities.
            effort: Optional efforts.
            external_torque: Optional external torques.

        Returns:
            joint_state_t instance.
        """

        ndof = len(name)

        def parse(x):
            """Helper to parse optional arrays or fill with zeros."""
            return np.asarray(x).flatten().tolist() if x is not None else np.zeros(ndof)

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
    """
    Add helpers to the `joy_t` message for accessing and constructing joystick data.

    Adds:
        - get_axis(), get_button(): Read individual joystick elements by name/index.
        - init(): Factory method to construct a complete joy_t instance.
    """

    # -------- read --------
    def get_axis(self, a: str | int | list[str] | list[int]):
        """Get joystick axis/axes by name(s) or index/indices."""
        return _get_named_item(self, "axis_names", "axes", a)

    def get_button(self, b: str | int | list[str] | list[int]):
        """Get joystick button(s) by name(s) or index/indices."""
        return _get_named_item(self, "button_names", "buttons", b)

    # -------- factory --------
    @classmethod
    def init(
        c,
        axis_names: list[str],
        axes: list[float] = [],
        button_names: list[str] = [],
        buttons: list[int] = [],
    ):
        """
        Factory for creating a joy_t message with named axes and buttons.

        Args:
            axis_names: List of axis names.
            axes: Corresponding axis values.
            button_names: List of button names.
            buttons: Corresponding button states.

        Returns:
            joy_t instance.
        """
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
def _add_image_helpers(cls):
    """
    Add array and image conversion utilities to the `image_t` message.

    Adds:
        - as_array(): Convert raw image data to a NumPy array.
        - as_image(): Convert raw data to a PIL.Image object.
        - from_array(): Factory to construct image_t from a NumPy array.
        - from_image(): Factory to construct image_t from a PIL.Image.
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

    def as_image(self) -> Image.Image:
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
    def from_image(c, img: Image.Image, *, frame_name: str = "") -> "image_t":
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
    cls.as_image = as_image
    cls.from_array = from_array
    cls.from_image = from_image
    return cls


# ----------------------------------------------------------------------
# helpers specific to laser_scan_helpers_t
# ----------------------------------------------------------------------
def _patch_laser_scan_with_arraylists(cls):
    """
    Patch the `laser_scan_t` class so that its `angles` and `ranges`
    fields are wrapped with a helper that provides `.as_array()`.

    This function:
    - Wraps `angles` and `ranges` with a subclass of list (`ArrayList`) that adds `.as_array()`.
    - Overrides the class-level `decode()` method to apply wrapping post-deserialization.
    - Exposes `_wrap_lists_post_decode()` for optional manual use.
    """

    # The fields that should be wrapped
    seq_fields = ("angles", "ranges")

    class ArrayList(list):
        """
        A list subclass that adds a convenient `.as_array()` method.

        This allows the user to call `.as_array()` on angles and ranges
        to retrieve them as NumPy arrays.
        """

        __slots__ = ()

        def as_array(self):
            """
            Return a copy of the list contents as a NumPy array.

            Returns:
                np.ndarray: Array of list values.
            """
            return np.asarray(self)

    # ------------------------------------------------------------------
    # 1. helper to (re)wrap an *instance* in‑place
    # ------------------------------------------------------------------
    def _wrap_lists(obj):
        """
        Replace fields in the object with `ArrayList` if applicable.

        Args:
            obj: The laser_scan_t instance to patch.

        Returns:
            The patched instance with wrapped fields.
        """
        for name in seq_fields:
            val = getattr(obj, name)
            # Accept list **or tuple**; avoid double‑wrapping
            if isinstance(val, (list, tuple)) and not isinstance(val, ArrayList):
                setattr(obj, name, ArrayList(val))
        return obj

    # ------------------------------------------------------------------
    # 2. patch the class‑level static decode() so every decode result
    #    is automatically wrapped before it leaves the function.
    # ------------------------------------------------------------------
    original_decode = cls.decode

    @staticmethod
    @wraps(original_decode)
    def decode(data):
        """
        Override for the class's decode method.

        Ensures that decoded instances have their fields wrapped
        with `ArrayList` for convenient usage.

        Args:
            data (bytes): The raw LCM data to decode.

        Returns:
            An instance of laser_scan_t with wrapped list fields.
        """
        return _wrap_lists(original_decode(data))

    cls.decode = decode  # override

    # ------------------------------------------------------------------
    # 3. expose the helper so factory methods (e.g. cls.init) can call
    #    it explicitly on freshly‑created objects.
    # ------------------------------------------------------------------
    cls._wrap_lists_post_decode = staticmethod(_wrap_lists)

    return cls  # allow decorator‑style use


def _add_laser_scan_helpers(cls):
    """
    Add an initialization helper for `laser_scan_t`.

    Adds:
        - init(): Factory method for setting angles and ranges.
    """

    # -------- factory --------
    @classmethod
    def init(c, angles: list[float] | np.ndarray, ranges: list[float] | np.ndarray):
        """Factory to create a `laser_scan_t` from angles and ranges."""
        if len(angles) != len(ranges):
            raise ValueError("angles and ranges must have the same length")

        obj = c()
        obj.angles = np.asarray(angles).flatten().tolist()
        obj.ranges = np.asarray(ranges).flatten().tolist()
        obj.n = len(obj.angles)

        obj = cls._wrap_lists_post_decode(obj)

        return obj

    # ------ patch class ------
    cls.init = init

    return cls


# ----------------------------------------------------------------------
# helpers specific to imu_t
# ----------------------------------------------------------------------
def _add_imu_helpers(cls):
    """
    Patch imu_t so you can (a) build one easily from arrays/Rotation, and
    (b) grab NumPy views of its nested orientation / angular_velocity /
    linear_acceleration fields.

    NOTE: We *do not* attach helpers to the member descriptors (cls.orientation,
    etc.) because those are slot descriptors without their own attribute
    namespace. Instead we add *methods on the imu_t class* that reach inward.
    """

    # --- generic -------------------------------------------------------------
    def _slots_to_array(obj) -> np.ndarray:
        # Works for any LCM struct w/ __slots__
        return np.array([getattr(obj, s) for s in obj.__slots__], dtype=float)

    # -------- read --------
    def orientation_as_array(self) -> np.ndarray:
        return _slots_to_array(self.orientation)

    def angular_velocity_as_array(self) -> np.ndarray:
        return _slots_to_array(self.angular_velocity)

    def linear_acceleration_as_array(self) -> np.ndarray:
        return _slots_to_array(self.linear_acceleration)

    # -------- factory --------
    @classmethod
    def init(
        c,
        orientation: list[float] | np.ndarray | Rot,
        angular_velocity: list[float] | np.ndarray,
        linear_acceleration: list[float] | np.ndarray,
    ):
        """
        Build an imu_t.

        *orientation* may be a scipy Rotation; it will be converted to quat
        in (x, y, z, w) or (w, x, y, z) order depending on the generated type's slots.
        We simply fill slots in order; caller must supply matching ordering if
        not using Rot.
        """
        # Convert orientation --------------------------------------------------
        if isinstance(orientation, Rot):
            # SciPy returns as (x, y, z, w)
            orientation = orientation.as_quat()  # ndarray, shape (4,)
        orientation = np.asarray(orientation, dtype=float).flatten()

        ang = np.asarray(angular_velocity, dtype=float).flatten()
        acc = np.asarray(linear_acceleration, dtype=float).flatten()

        obj = c()

        # Helper to stuff an array into an LCM sub-struct in slot order --------
        def _fill(subobj, arr, name):
            slots = getattr(subobj, "__slots__", None)
            if slots is None:
                raise AttributeError(f"{name} has no __slots__ (unexpected)")
            if len(arr) != len(slots):
                raise ValueError(f"{name} expects {len(slots)} values; got {len(arr)}")
            for s, v in zip(slots, arr):
                setattr(subobj, s, float(v))

        _fill(obj.orientation, orientation, "orientation")
        _fill(obj.angular_velocity, ang, "angular_velocity")
        _fill(obj.linear_acceleration, acc, "linear_acceleration")

        return obj

    # ------ patch class ------
    cls.orientation_as_array = orientation_as_array
    cls.angular_velocity_as_array = angular_velocity_as_array
    cls.linear_acceleration_as_array = linear_acceleration_as_array
    cls.init = init

    return cls


# ----------------------------------------------------------------------
# patch the raw classes in-place
# ----------------------------------------------------------------------
_add_joint_state_helpers(_joint_state_t)
_add_joy_helpers(_joy_t)
_add_image_helpers(_image_t)
_add_laser_scan_helpers(_laser_scan_t)
_patch_laser_scan_with_arraylists(_laser_scan_t)
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
    "joy_t",
    "image_t",
    "laser_scan_t",
    "imu_t",
]

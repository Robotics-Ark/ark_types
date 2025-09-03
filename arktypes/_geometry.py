from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# ----------------------------------------------------------------------
# bring the raw message classes into the namespace
# ----------------------------------------------------------------------
from _arktypes_geometry.vector3_t import vector3_t as _vector3_t
from _arktypes_geometry.quaternion_t import quaternion_t as _quaternion_t
from _arktypes_geometry.transform_t import transform_t as _transform_t
from _arktypes_geometry.wrench_t import wrench_t as _wrench_t
from _arktypes_geometry.twist_t import twist_t as _twist_t


# ----------------------------------------------------------------------
# generic helpers for any fixed‑size numeric struct
# ----------------------------------------------------------------------
def _add_array_helpers(cls):
    """
    Add generic helpers `.as_array()` and `.from_array()` to fixed-size numeric structs.

    `.as_array()` returns a list of float values in the order of class slots.
    `.from_array()` initializes an instance from a sequence of float values.

    Args:
        cls: The class to modify in-place.

    Returns:
        The modified class.
    """
    slots = cls.__slots__

    def as_array(self) -> list[float]:
        """
        Convert the struct to a list of floats.

        Returns:
            list[float]: Values of all fields in order.
        """
        return [getattr(self, name) for name in slots]

    @classmethod
    def from_array(c, seq) -> "c":
        """
        Construct a new instance from a list or array of float values.

        Args:
            seq (Sequence[float]): Input values matching the number of struct fields.

        Returns:
            An instance of the struct with values assigned.

        Raises:
            ValueError: If input does not match the expected size.
        """
        if len(seq) != len(slots):
            raise ValueError(
                f"{c.__name__}.from_array() expects {len(slots)} values, got {len(seq)}"
            )
        obj = c()
        for name, value in zip(slots, seq):
            setattr(obj, name, float(value))
        return obj

    cls.as_array = as_array
    cls.from_array = from_array

    return cls


# ----------------------------------------------------------------------
# helpers specific to vector3_t
# ----------------------------------------------------------------------
def _add_vector3_helpers(cls):
    """
    Add vector-specific helper `.identity()` to `vector3_t`.

    `.identity()` returns a default zero vector.

    Args:
        cls: The vector3_t class.

    Returns:
        The modified class.
    """

    @classmethod
    def identity(c):
        """
        Return the identity (zero) vector.

        Returns:
            vector3_t: A vector with all components zero.
        """
        return c()

    cls.identity = identity

    return cls


# ----------------------------------------------------------------------
# helpers specific to quaternion_t
# ----------------------------------------------------------------------
def _add_rotation_helpers(cls):
    """
    Add rotation-related helpers to `quaternion_t` class.

    Adds:
        - `.as_rotation()` to get a `scipy` Rotation object.
        - `.set_from_rotation()` to mutate an instance from rotation.
        - `.from_rotation()` to create a new instance from rotation.
        - `.identity()` to return a unit quaternion.

    Args:
        cls: The quaternion_t class.

    Returns:
        The modified class.
    """

    def as_rotation(self) -> Rot:
        """
        Convert quaternion to a `scipy.spatial.transform.Rotation`.

        Returns:
            Rot: Rotation instance representing the quaternion.
        """
        return Rot.from_quat([self.x, self.y, self.z, self.w])

    def set_from_rotation(self, rot: Rot | np.ndarray):
        """
        Set the quaternion values from a rotation object or 3x3 matrix.

        Args:
            rot (Rot | np.ndarray): A `Rotation` object or a 3x3 ndarray.

        Returns:
            The modified instance.

        Raises:
            ValueError: If ndarray is not 3x3.
            TypeError: If input is of unsupported type.
        """
        if isinstance(rot, Rot):
            quat = rot.as_quat()
        elif isinstance(rot, np.ndarray):
            if rot.shape != (3, 3):
                raise ValueError(
                    f"Expected rotation matrix with shape (3,3); got {rot.shape}"
                )
            quat = Rot.from_matrix(rot).as_quat()
        else:
            raise TypeError(
                "set_from_rotation() expects a scipy Rotation or a (3,3) ndarray"
            )

        self.x, self.y, self.z, self.w = map(float, quat)
        return self

    @classmethod
    def from_rotation(c, rot: Rot | np.ndarray):
        """
        Construct a quaternion_t instance from a rotation object or matrix.

        Args:
            rot (Rot | np.ndarray): A `Rotation` or a 3x3 rotation matrix.

        Returns:
            quaternion_t: A new quaternion instance.
        """
        return c().set_from_rotation(rot)

    @classmethod
    def identity(c):
        """
        Return the identity rotation (no rotation).

        Returns:
            quaternion_t: A unit quaternion representing no rotation.
        """
        return c.from_rotation(np.eye(3))

    cls.as_rotation = as_rotation
    cls.set_from_rotation = set_from_rotation
    cls.from_rotation = from_rotation
    cls.identity = identity

    return cls


# ----------------------------------------------------------------------
# helpers specific to transform_t
# ----------------------------------------------------------------------
def _add_transform_helpers(cls):
    """
    Add helpers for `transform_t` class for conversion to/from arrays.

    Adds:
        - `.as_array()` to convert to 4x4 homogeneous matrix.
        - `.from_array()` to initialize from 4x4 matrix.
        - `.identity()` to return identity transform.

    Args:
        cls: The transform_t class.

    Returns:
        The modified class.
    """

    def as_array(self) -> np.ndarray:
        """
        Convert the transform to a 4x4 homogeneous transformation matrix.

        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        tf = np.eye(4)
        tf[:3, 3] = self.translation.as_array()
        tf[:3, :3] = self.rotation.as_rotation().as_matrix()
        return tf

    @classmethod
    def from_array(c, tf: np.ndarray):
        """
        Construct a `transform_t` instance from a 4x4 matrix.

        Args:
            tf (np.ndarray): A 4x4 transformation matrix.

        Returns:
            transform_t: A new instance with parsed translation and rotation.
        """
        obj = c()
        obj.translation = vector3_t.from_array(tf[:3, 3])
        obj.rotation = quaternion_t.from_rotation(tf[:3, :3])
        return obj

    @classmethod
    def identity(c):
        """
        Return an identity transform (no translation, no rotation).

        Returns:
            transform_t: The identity transformation.
        """
        return c.from_array(np.eye(4))

    cls.as_array = as_array
    cls.from_array = from_array
    cls.identity = identity

    return cls


# ----------------------------------------------------------------------
# helpers specific to twist_t
# ----------------------------------------------------------------------
def _add_twist_helpers(cls):
    """
    Add helpers to `twist_t` for converting to/from arrays.

    Adds:
        - `.as_array()` to serialize linear and angular velocities.
        - `.from_array()` to construct from an array of size 6.

    Args:
        cls: The twist_t class.

    Returns:
        The modified class.
    """

    def as_array(self, linear_first: bool = True):
        """
        Convert twist to a 6-element array.

        Args:
            linear_first (bool): Whether linear velocity comes before angular.

        Returns:
            np.ndarray: Combined 6-element vector.
        """
        l = self.linear.as_array()
        a = self.angular.as_array()
        return np.concatenate((l, a) if linear_first else (a, l))

    @classmethod
    def from_array(c, array, linear_first: bool = True):
        """
        Construct a `twist_t` from a 6-element array.

        Args:
            array (Sequence[float]): 6-element array.
            linear_first (bool): Whether linear velocity is first in the array.

        Returns:
            twist_t: The constructed twist object.

        Raises:
            ValueError: If input is not length 6.
        """
        if len(array) != 6:
            raise ValueError(f"array must be length 6, got {len(array)}")
        if linear_first:
            l, a = array[:3], array[3:6]
        else:
            a, l = array[:3], array[3:6]
        obj = c()
        obj.linear = vector3_t.from_array(l)
        obj.angular = vector3_t.from_array(a)
        return obj

    cls.as_array = as_array
    cls.from_array = from_array

    return cls


# ----------------------------------------------------------------------
# helpers specific to wrench_t
# ----------------------------------------------------------------------
def _add_wrench_helpers(cls):
    """
    Add helpers to `wrench_t` for converting to/from arrays.

    Adds:
        - `.as_array()` to serialize force and torque.
        - `.from_array()` to construct from a 6-element array.

    Args:
        cls: The wrench_t class.

    Returns:
        The modified class.
    """

    def as_array(self, force_first: bool = True):
        """
        Convert wrench to a 6-element array.

        Args:
            force_first (bool): Whether force comes before torque.

        Returns:
            np.ndarray: Combined 6-element force-torque vector.
        """
        f = self.force.as_array()
        t = self.torque.as_array()
        return np.concatenate((f, t) if force_first else (t, f))

    @classmethod
    def from_array(c, array, force_first: bool = True):
        """
        Construct a `wrench_t` from a 6-element array.

        Args:
            array (Sequence[float]): 6-element array.
            force_first (bool): Whether force is first in the array.

        Returns:
            wrench_t: The constructed wrench object.

        Raises:
            ValueError: If input is not length 6.
        """
        if len(array) != 6:
            raise ValueError(f"array must be length 6, got {len(array)}")
        if force_first:
            f, t = array[:3], array[3:6]
        else:
            t, f = array[:3], array[3:6]

        obj = c()
        obj.force = vector3_t.from_array(f)
        obj.torque = vector3_t.from_array(t)

        return obj

    cls.as_array = as_array
    cls.from_array = from_array

    return cls


# ----------------------------------------------------------------------
# patch the raw classes in‑place
# ----------------------------------------------------------------------
_add_array_helpers(_vector3_t)
_add_vector3_helpers(_vector3_t)

_add_array_helpers(_quaternion_t)
_add_rotation_helpers(_quaternion_t)

_add_transform_helpers(_transform_t)

_add_twist_helpers(_twist_t)

_add_wrench_helpers(_wrench_t)

# ----------------------------------------------------------------------
# re‑export under user‑friendly names
# ----------------------------------------------------------------------
vector3_t = _vector3_t
quaternion_t = _quaternion_t
transform_t = _transform_t
twist_t = _twist_t
wrench_t = _wrench_t

__all__ = [
    "vector3_t",
    "quaternion_t",
    "transform_t",
    "twist_t",
    "wrench_t",
]

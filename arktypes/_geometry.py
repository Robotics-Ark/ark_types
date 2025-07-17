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
    """Attach .as_array() and .from_array()."""
    slots = cls.__slots__

    def as_array(self) -> list[float]:
        return [getattr(self, name) for name in slots]

    @classmethod
    def from_array(c, seq) -> "c":  # type: ignore[name-defined]
        if len(seq) != len(slots):
            raise ValueError(
                f"{c.__name__}.from_array() expects {len(slots)} values, got {len(seq)}"
            )
        obj = c()
        for name, value in zip(slots, seq):
            setattr(obj, name, float(value))
        return obj

    cls.as_array = as_array  # type: ignore[attr-defined]
    cls.from_array = from_array  # type: ignore[attr-defined]
    return cls


# ----------------------------------------------------------------------
# helpers specific to vector3_t
# ----------------------------------------------------------------------
def _add_vector3_helpers(cls):

    # -------- factory --------
    @classmethod
    def identity(c):
        """Returns the identity vector."""
        return c()

    cls.identity = identity  # type: ignore[attr-defined]

    return cls


# ----------------------------------------------------------------------
# helpers specific to quaternion_t
# ----------------------------------------------------------------------
def _add_rotation_helpers(cls):
    """Attach .as_rotation(), .set_from_rotation() and .from_rotation()."""

    # -------- read --------
    def as_rotation(self) -> Rot:
        return Rot.from_quat([self.x, self.y, self.z, self.w])

    # -------- write (mutating) --------
    def set_from_rotation(self, rot: Rot | np.ndarray):
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

    # -------- factory --------
    @classmethod
    def from_rotation(c, rot: Rot | np.ndarray):
        """Return a **new** quaternion_t built from `rot`."""
        return c().set_from_rotation(rot)

    @classmethod
    def identity(c):
        """Returns the identity rotation."""
        return c.from_rotation(np.eye(3))

    cls.as_rotation = as_rotation  # type: ignore[attr-defined]
    cls.set_from_rotation = set_from_rotation  # type: ignore[attr-defined]
    cls.from_rotation = from_rotation  # type: ignore[attr-defined]
    cls.identity = identity  # type: ignore[attr-defined]

    return cls


# ----------------------------------------------------------------------
# helpers specific to transform_t
# ----------------------------------------------------------------------
def _add_transform_helpers(cls):
    """Attach .as_array()."""

    # -------- read --------
    def as_array(self) -> np.ndarray:
        tf = np.eye(4)
        tf[:3, 3] = self.translation.as_array()
        tf[:3, :3] = self.rotation.as_rotation().as_matrix()
        return tf

    # -------- factory --------
    @classmethod
    def from_array(c, tf: np.ndarray):
        """Return a **new** transform_t from `ndarray`."""
        obj = c()
        obj.translation = vector3_t.from_array(tf[:3, 3])
        obj.rotation = quaternion_t.from_rotation(tf[:3, :3])
        return obj

    @classmethod
    def identity(c):
        """Returns the identity transform."""
        return c.from_array(np.eye(4))

    cls.as_array = as_array  # type: ignore[attr-defined]
    cls.from_array = from_array  # type: ignore[attr-defined]
    cls.identity = identity  # type: ignore[attr-defined]

    return cls


# ----------------------------------------------------------------------
# helpers specific to twist_t
# ----------------------------------------------------------------------
def _add_twist_helpers(cls):

    # -------- read --------
    def as_array(self, linear_first: bool = True):
        l = self.linear.as_array()
        a = self.angular.as_array()
        return np.concatenate((l, a) if linear_first else (a, l))

    # -------- factory --------
    @classmethod
    def from_array(c, array, linear_first: bool = True):
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

    cls.as_array = as_array  # type: ignore[attr-defined]
    cls.from_array = from_array  # type: ignore[attr-defined]

    return cls


# ----------------------------------------------------------------------
# helpers specific to wrench_t
# ----------------------------------------------------------------------
def _add_wrench_helpers(cls):

    # -------- read --------
    def as_array(self, force_first: bool = True):
        l = self.linear.as_array()
        a = self.angular.as_array()
        return np.concatenate((l, a) if linear_first else (a, l))

    # -------- factory --------
    @classmethod
    def from_array(c, array, linear_first: bool = True):
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

    cls.as_array = as_array  # type: ignore[attr-defined]
    cls.from_array = from_array  # type: ignore[attr-defined]

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
]

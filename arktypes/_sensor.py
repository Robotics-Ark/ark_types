from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# ----------------------------------------------------------------------
# bring the raw message classes into the namespace
# ----------------------------------------------------------------------
from _arktypes_sensor.joint_state_t import joint_state_t as _joint_state_t
from _arktypes_sensor.joy_t import joy_t as _joy_t
from _arktypes_sensor.image_t import image_t as _image_t
from _arktypes_sensor.rgbd_t import rgbd_t as _rgbd_t
from _arktypes_sensor.laser_scan_t import laser_scan_t as _laser_scan_t
from _arktypes_sensor.imu_t import imu_t as _imu_t


# ----------------------------------------------------------------------
# helpers specific to joint_state_t
# ----------------------------------------------------------------------
def _add_joint_state_helpers(cls):

    # -------- read --------
    def _get_item(self, attr: str, x: str | int | list[str] | list[int]):
        if isinstance(x, int):
            arr = getattr(self, attr)
            return arr[p]
        elif isinstance(p, str):
            return _get_item(self, attr, self.name.index(p))
        elif isinstance(p, list):
            if isinstance(p[0], (int, str)):
                return [_get_item(i) for i in p]
            else:
                raise ValueError(
                    "when x is a list then the elements should all be either 'str' or 'int', "
                    f"got instead '{type(p[0])}'"
                )
        else:
            raise ValueError(f"x must be a 'str' or 'int', got {type(x)}")

    def get_position(self, p: str | int):
        return _get_item(self, "position", p)

    def get_velocity(self, v: str | int):
        return _get_item(self, "velocity", v)

    def get_effort(self, e: str | int):
        return _get_item(self, "effort", e)

    def get_external_torque(self, e: str | int):
        return _get_item(self, "external_torque", e)

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

    # -------- factory --------
    @classmethod
    def init(c, axes: list[float] = [], buttons: list[int] = []):
        obj = c()
        obj.naxes = len(axes)
        obj.axes = axes
        obj.nbuttons = len(buttons)
        obj.buttons = buttons
        return obj

    return cls


# ----------------------------------------------------------------------
# helpers specific to image_t
# ----------------------------------------------------------------------
def _add_image_helpers(cls):
    # TODO
    return cls


# ----------------------------------------------------------------------
# helpers specific to rgbd_t
# ----------------------------------------------------------------------
def _add_rgbd_helpers(cls):
    # TODO
    return cls


# ----------------------------------------------------------------------
# helpers specific to rgbd_t
# ----------------------------------------------------------------------
def _add_laser_scan_helpers(cls):
    # TODO
    return cls


# ----------------------------------------------------------------------
# helpers specific to imu_t
# ----------------------------------------------------------------------
def _add_imu_helpers(cls):
    # TODO
    return cls


# ----------------------------------------------------------------------
# patch the raw classes in‑place
# ----------------------------------------------------------------------
_add_joint_state_helpers(_joint_state_t)
_add_joy_helpers(_joy_t)
_add_image_helpers(_image_t)
_add_rgbd_helpers(_rgbd_t)
_add_laser_scan_helpers(_laser_scan_t)
_add_imu_helpers(_imu_t)

# ----------------------------------------------------------------------
# re‑export under user‑friendly names
# ----------------------------------------------------------------------
joint_state_t = _joint_state_t
joy_t = _joy_t
image_t = _image_t
rgbd_t = _rgbd_t
laser_scan_t = _laser_scan_t
imu_t = _imu_t

__all__ = [
    "joint_state_t",
    "image_t",
    "rgbd_t",
    "laser_scan_t",
    "imu_t",
]

from typing import Any
import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation as Rot

####################################################################################
# Import all messages from geometry module and add subclasses with helper methods
#
from _arktypes_geometry.quaternion_t import quaternion_t as _quaternion_t
from _arktypes_geometry.vector3_t import vector3_t as _vector3_t
from _arktypes_geometry.transform_t import transform_t as _transform_t
from _arktypes_geometry.twist_t import twist_t as _twist_t
from _arktypes_geometry.wrench_t import wrench_t as _wrench_t


class quaternion_t(_quaternion_t):

    @staticmethod
    def decode(data: bytes):
        q = _quaternion_t.decode(data)
        return quaternion_t.from_array([q.x, q.y, q.z, q.w])

    @classmethod
    def from_rotation(cls, rotation: Rot):
        """Create a quaternion_t from a scipy Rotation object."""
        return cls.from_array(rotation.as_quat())

    def as_rotation(self) -> Rot:
        """Convert quaternion_t to a scipy Rotation object."""
        return Rot.from_quat([self.x, self.y, self.z, self.w])

    @classmethod
    def from_array(cls, array: list | tuple | ndarray):
        q = cls()
        q.x, q.y, q.z, q.w = array
        return q

    def as_array(self) -> ndarray:
        """Convert quaternion_t to an array."""
        return np.array([self.x, self.y, self.z, self.w])

    @classmethod
    def identity(cls):
        """Set quaternion to identity (0, 0, 0, 1)."""
        q = cls()
        q.x, q.y, q.z, q.w = 0.0, 0.0, 0.0, 1.0
        return q


class vector3_t(_vector3_t):

    @staticmethod
    def decode(data: bytes):
        v = _vector3_t.decode(data)
        return vector3_t.from_array([v.x, v.y, v.z])

    @classmethod
    def from_array(cls, array: list | tuple | ndarray):
        v = cls()
        v.x, v.y, v.z = array
        return v

    def as_array(self) -> ndarray:
        """Convert vector3_t to a list."""
        return np.array([self.x, self.y, self.z])

    @classmethod
    def identity(cls):
        """Set vector3_t to zero (0, 0, 0)."""
        v = cls()
        v.x, v.y, v.z = 0.0, 0.0, 0.0
        return v


class transform_t(_transform_t):

    @staticmethod
    def decode(data: bytes):
        _t = _transform_t.decode(data)
        t = transform_t()
        tr, ro = _t.translation, _t.rotation
        t.translation = vector3_t.from_array([tr.x, tr.y, tr.z])
        t.rotation = quaternion_t.from_array([ro.x, ro.y, ro.z, ro.w])
        return t

    @classmethod
    def from_arrays(
        cls,
        translation: list | tuple | ndarray | None = None,
        rotation: list | tuple | ndarray | Rot | None = None,
    ):
        """Create a transform_t from translation and rotation arrays."""
        t = cls()
        t.translation = vector3_t.from_array(translation)
        if isinstance(rotation, Rot):
            t.rotation = quaternion_t.from_rotation(rotation)
        elif rotation is None:
            t.rotation = quaternion_t.identity()
        else:
            if len(rotation) != 4:
                raise ValueError(
                    "Rotation must be a 4-element array or a Rotation object."
                )
            t.rotation = quaternion_t.from_array(rotation)
        return t

    def as_arrays(self):
        """Convert transform_t to translation and rotation arrays."""
        return (
            self.translation.as_array(),
            self.rotation.as_array(),
        )

    @classmethod
    def from_array(cls, tf: ndarray):
        """Create a transform_t from a 4x4 transformation matrix."""
        if tf.shape != (4, 4):
            raise ValueError("Input must be a 4x4 transformation matrix.")

        t = cls()
        t.translation = vector3_t.from_array(tf[:3, 3])
        rot = Rot.from_matrix(tf[:3, :3])
        t.rotation = quaternion_t.from_rotation(rot)
        return t

    def as_array(self) -> ndarray:
        """Convert transform_t to a 4x4 transformation matrix."""
        tf = np.eye(4)
        tf[:3, :3] = self.rotation.as_rotation().as_matrix()
        tf[:3, 3] = self.translation.as_array()
        return tf

    @classmethod
    def identity(cls):
        """Set transform to identity (translation: 0, 0, 0; rotation: identity quaternion)."""
        t = cls()
        t.translation = vector3_t.identity()
        t.rotation = quaternion_t.identity()
        return t


class twist_t(_twist_t):

    @staticmethod
    def decode(data: bytes):
        t = _twist_t.decode(data)
        lin, ang = t.linear, t.angular
        t.linear = vector3_t.from_array([lin.x, lin.y, lin.z])
        t.angular = vector3_t.from_array([ang.x, ang.y, ang.z])
        return t

    @classmethod
    def from_arrays(
        cls,
        linear: list | tuple | ndarray | None = None,
        angular: list | tuple | ndarray | None = None,
    ):
        """Create a twist_t from linear and angular arrays."""
        t = cls()
        t.linear = vector3_t.from_array(linear)
        t.angular = vector3_t.from_array(angular)
        return t

    def as_arrays(self):
        """Convert twist_t to linear and angular arrays."""
        return (
            self.linear.as_array(),
            self.angular.as_array(),
        )


class wrench_t(_wrench_t):

    @staticmethod
    def decode(data: bytes):
        w = _wrench_t.decode(data)
        force, torque = w.force, w.torque
        w.force = vector3_t.from_array([force.x, force.y, force.z])
        w.torque = vector3_t.from_array([torque.x, torque.y, torque.z])
        return w

    @classmethod
    def from_arrays(
        cls,
        force: list | tuple | ndarray | None = None,
        torque: list | tuple | ndarray | None = None,
    ):
        """Create a wrench_t from force and torque arrays."""
        w = cls()
        w.force = vector3_t.from_array(force)
        w.torque = vector3_t.from_array(torque)
        return w

    def as_arrays(self):
        """Convert wrench_t to force and torque arrays."""
        return (
            self.force.as_array(),
            self.torque.as_array(),
        )


####################################################################################
# Import all messages from info module and add subclasses with helper methods
#
from _arktypes_info.comms_info_t import comms_info_t as _comms_info_t
from _arktypes_info.listener_info_t import listener_info_t as _listener_info_t
from _arktypes_info.network_info_t import network_info_t as _network_info_t
from _arktypes_info.node_info_t import node_info_t as _node_info_t
from _arktypes_info.publisher_info_t import publisher_info_t as _publisher_info_t
from _arktypes_info.service_info_t import service_info_t as _service_info_t
from _arktypes_info.subscriber_info_t import subscriber_info_t as _subscriber_info_t


class comms_info_t(_comms_info_t):

    @staticmethod
    def decode(data: bytes):
        info = _comms_info_t.decode(data)
        return comms_info_t.from_dict(info.to_dict())

    @classmethod
    def from_lists(
        cls,
        listeners: list[dict[str, Any]],
        subscribers: list[dict[str, Any]],
        publishers: list[dict[str, Any]],
        services: list[dict[str, Any]],
    ):
        info = cls()
        info.n_listeners = len(listeners)
        info.listeners = [listener_info_t.from_dict(l) for l in listeners]
        info.n_subscribers = len(subscribers)
        info.subscribers = [subscriber_info_t.from_dict(s) for s in subscribers]
        info.n_publishers = len(publishers)
        info.publishers = [publisher_info_t.from_dict(p) for p in publishers]
        info.n_services = len(services)
        info.services = [service_info_t.from_dict(s) for s in services]
        return info


####################################################################################
# Import all messages from robot module and add subclasses with helper methods
#

from _arktypes_robot.joint_state_t import joint_state_t as _joint_state_t
from _arktypes_robot.joint_state_stamped_t import (
    joint_state_stamped_t as _joint_state_stamped_t,
)


class joint_state_t(_joint_state_t):

    @staticmethod
    def decode(data: bytes):
        _js = _joint_state_t.decode(data)
        js = joint_state_t()
        js.ndof = _js.ndof
        js.name = _js.name
        js.position = _js.position
        js.velocity = _js.velocity
        js.effort = _js.effort
        js.ext_torque = _js.ext_torque
        return js

    @classmethod
    def from_lists(
        cls,
        name: str,
        position: list[float],
        velocity: list[float] = None,
        effort: list[float] = None,
        ext_torque: list[float] = None,
    ):
        js = cls()
        js.ndof = len(position)
        js.name = name
        js.position = position
        js.velocity = velocity if velocity is not None else [0.0] * js.ndof
        js.effort = effort if effort is not None else [0.0] * js.ndof
        js.ext_torque = ext_torque if ext_torque is not None else [0.0] * js.ndof
        return js

    def get_position(self, name: str | list[str]) -> float | list[float]:
        """Get joint position by name or return all positions."""
        if isinstance(name, str):
            if name in self.name:
                idx = self.name.index(name)
                return self.position[idx]
            else:
                raise ValueError(f"Joint '{name}' not found in state.")
        elif isinstance(name, list):
            return [self.get_position(n) for n in name]
        else:
            raise TypeError("name must be a string or a list of strings.")


class joint_state_stamped_t(_joint_state_stamped_t):

    @staticmethod
    def decode(data: bytes):
        _js = _joint_state_stamped_t.decode(data)
        js = joint_state_stamped_t()
        js.header = _js.header
        js.joint_state = joint_state_t.from_lists(
            _js.name,
            _js.position,
            _js.velocity,
            _js.effort,
            _js.ext_torque,
        )
        return js


####################################################################################
# Import all messages from sensor module and add subclasses with helper methods
#

# TODO


import cv2
import zlib
import numpy as np
from typing import Tuple, List, Dict

from arktypes import (
    float_t,
    float_vector_t, 
    float_array_t,
    image_t,
    rgbd_t, 
    laser_scan_t, 
    pose_2d_t, 
    velocity_2d_t,
    joint_group_command_t, 
    grid_config_t,
    wheel_config_t
)

def unpack_float(msg: float_t) -> float:
    """!
    Unpack a float_t message into a float value.

    @param msg  A float_t message containing a single float value.
    @return The unpacked float value.
    """
    return msg.data

def unpack_float_vector(msg: float_vector_t) -> np.ndarray:
    """!
    Unpack a float_vector_t message into a NumPy array.

    @param msg  A float_vector_t message containing float data.
    @return A NumPy array containing the unpacked float values.
    """
    data = msg.data
    data = np.array(data)
    return data

def unpack_float_array(msg: float_array_t) -> np.ndarray:
    """!
    Unpack a float_array_t message into a 2D NumPy array.

    @param msg  A float_array_t message containing float data.
    @return A NumPy array containing the unpacked float values.
    """
    data = msg.data
    data = np.array(data)
    return data

def unpack_image(msg: image_t) -> np.ndarray:
    """!
    Unpacks a serialized image_t message into an OpenCV-compatible NumPy array.

    Handles both compressed (JPEG/PNG) and uncompressed image data.

    @param msg The image_t message to be unpacked.
    @return A NumPy array representing the image, or None if unpacking fails.
    """
    img_data = np.frombuffer(msg.data, dtype=np.uint8)

    # Handle compression
    if msg.compression_method in (image_t.COMPRESSION_METHOD_JPEG, image_t.COMPRESSION_METHOD_PNG):
        # Decompress image
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decompress image")
            return
    elif msg.compression_method == image_t.COMPRESSION_METHOD_NOT_COMPRESSED:
        # Determine the number of channels based on pixel_format
        try:
            nchannels = num_channels[msg.pixel_format]
        except KeyError:
            print("Unsupported pixel format")
            return

        # Reshape the data to the original image dimensions
        try:
            img = img_data.reshape((msg.height, msg.width, nchannels))
        except ValueError as e:
            print(f"Error reshaping image data: {e}")
            return
    else:
        print("Unsupported compression method")
        return
    return img

def unpack_depth(msg: image_t) -> np.ndarray:
    """!
    Unpacks a compressed depth image from an image_t message into a NumPy array.

    The function expects the depth data to be zlib-compressed and stored as 16-bit unsigned integers
    representing millimeters. It converts the result to floating-point meters.

    @param msg The image_t message containing compressed depth data.
    @return A 2D NumPy array of depth values in meters.
    """
    depth_data = zlib.decompress(msg.data)
    depth = np.frombuffer(depth_data, dtype=np.uint16)
    depth = depth.astype(np.float32) / 1000
    depth = depth.reshape((msg.height, msg.width))
    return depth

def unpack_rgbd(msg: rgbd_t) -> Tuple[np.ndarray, np.ndarray]:
    """!
    Unpacks an rgbd_t message into separate RGB image and depth map arrays.

    This function extracts and decodes the image and depth components from an RGB-D message,
    returning them as OpenCV-compatible NumPy arrays.

    @param msg The rgbd_t message containing packed image and depth data.
    @return A tuple (image, depth), where:
        - image: A NumPy array representing the RGB image.
        - depth: A 2D NumPy array representing the depth map in meters.
    """
    image = unpack_image(msg.image)
    depth = unpack_depth(msg.depth)
    return image, depth


def unpack_laser_scan(msg: laser_scan_t) -> Tuple[np.ndarray, np.ndarray]:
    """!
    Unpack a laser_scan_t message into separate NumPy arrays for angles and ranges.

    @param msg  A laser_scan_t message with packed angle and range data.
    @return A tuple of two NumPy arrays: (angles, ranges).
    """
    angles = unpack_float_vector(msg.angles)
    ranges = unpack_float_vector(msg.ranges)
    return angles, ranges

def unpack_pose_2d(msg: pose_2d_t) -> Tuple[float, float, float]:
    """!
    Unpack a pose_2d_t message into individual pose components.

    @param msg  A pose_2d_t message containing x, y, and theta values.
    @return A tuple (x, y, theta) representing the 2D pose.
    """
    x = msg.x
    y = msg.y
    theta = msg.theta
    return x, y, theta

def unpack_velocity_2d(msg: velocity_2d_t) -> tuple[float, float]:
    """!
    Unpack a velocity_2d_t message into linear and angular velocity components.

    @param msg: The velocity_2d_t message to unpack.
    @return A tuple containing (linear_velocity, angular_velocity).
    """
    linear_velocity = msg.linear
    angular_velocity = msg.angular
    return linear_velocity, angular_velocity

def unpack_joint_group_command(msg: joint_group_command_t) -> Tuple[List, str]:
    """!
    Unpacks a joint group command into the values and the name.

    @param msg  A joint_ground_command_t message
    @return A tuple (cmd, name) representing the command values and the group name
    """
    cmd = msg.cmd
    name = msg.name
    return cmd, name

def unpack_grid_config(msg: grid_config_t) -> Tuple[Dict[str, List[float]], float]:
    """!
    Unpack a grid_config_t message into scene bounds and grid size.

    @param msg  The grid_config_t message to unpack.
    @return A tuple containing:
            - scene_bounds: A dictionary with 'x' and 'y' keys mapping to the corresponding [min, max] bounds.
            - grid_size: The size of each grid cell.
    """
    scene_bounds = {
        "x": msg.x_bounds,
        "y": msg.y_bounds
    }
    grid_size = msg.grid_size
    return scene_bounds, grid_size

def unpack_wheel_config(msg: wheel_config_t) -> Tuple:
    """!
    Unpacks a wheel configuration message.
    @param msg The wheel configuration message of type wheel_config_t.
    @return A tuple containing:
        - radius:     Radius of the wheels.
        - thread:     Distance between the wheels
    """
    radius = msg.radius
    thread = msg.thread
    return radius, thread

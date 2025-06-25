
import cv2
import zlib
import array
import numpy as np
from typing import List, Dict

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

def pack_float(data: float) -> float_t:
    """!
    Pack a single float data into a float_t message.

    @param data  A float value to be packed.
    @return A float_t message with the data field populated.
    """
    msg = float_t()
    msg.data = data
    return msg

def pack_float_vector(data: np.ndarray) -> float_vector_t:
    """!
    Pack a list or array of float values into a float_vector_t message.

    @param data  A 1D NumPy array of float values.
    @return A float_vector_t message with populated data and size fields.
    """
    msg = float_vector_t()
    msg.n = len(data)
    msg.data = data
    return msg

def pack_float_array(data: np.ndarray) -> float_vector_t:
    """!
    Pack a 2D NumPy array of float values into a float_array_t message.

    @param data  A 2D NumPy array of shape (m, n) representing float values.
    @return A float_array_t message with the `m`, `n`, and `data` fields populated accordingly.
    """
    msg = float_array_t()
    msg.m = data.shape[0]
    msg.n = data.shape[1]
    msg.data = data
    return msg

def pack_image(image: np.ndarray, name: str = "") -> image_t:
    """!
    Converts an BGR image (as a NumPy array) into a serialized image_t message with compression.

    @param image The BGR image (NumPy array) to pack.
    @param name An optional name used to generate the frame name.
    @return An image_t message containing the encoded image and metadata, or None if encoding fails.
    """
    msg = image_t()

    # Fill in timestamp and frame_name
    msg.frame_name = f"{name}_frame"

    # Get image dimensions
    height, width, _ = image.shape
    msg.height = height
    msg.width = width

    # Set pixel format and channel type
    msg.pixel_format = image_t.PIXEL_FORMAT_BGR  # OpenCV uses BGR format
    msg.channel_type = image_t.CHANNEL_TYPE_UINT8  # OpenCV images are uint8

    # Set bigendian
    msg.bigendian = False

    # Set row_stride
    msg.row_stride = image.strides[0]  # Number of bytes per row

    # Compress the image using the selected method
    success, encoded_image = cv2.imencode('.png', image)
    if success:
        data = array.array('B', encoded_image.tobytes())
        msg.data = data
        # Set size of data
        msg.size = len(msg.data)
        # Set compression method
        msg.compression_method = image_t.COMPRESSION_METHOD_PNG
    else:
        log.warn("Failed to compress image")
        return None
    
    return msg

def pack_depth(depth: np.ndarray, name: str = "") -> image_t:
    """!
    Converts a depth image (as a NumPy array) into a compressed image_t message using zlib compression.

    @param depth A 2D NumPy array containing depth values in meters.
    @param name An optional name used to generate the frame name.
    @return An image_t message containing the compressed depth image and metadata.
    """
    msg = image_t()

    # Fill in timestamp and frame_name
    msg.frame_name = f"{name}_frame"

    # Get image dimensions
    height, width = depth.shape
    msg.height = height
    msg.width = width

    # Set pixel format and channel type
    msg.pixel_format = image_t.PIXEL_FORMAT_DEPTH  # OpenCV uses BGR format
    msg.channel_type = image_t.CHANNEL_TYPE_UINT16  # OpenCV images are uint8

    # Set bigendian
    msg.bigendian = False

    # Set row_stride
    msg.row_stride = depth.strides[0]  # Number of bytes per row

    # Compress the depth using zlib
    depth = (depth * 1000).astype(np.uint16) # Convert to mm
    depth_bytes = depth.tobytes()  # Convert depth map to raw bytes
    compressed_depth = zlib.compress(depth_bytes)

    # Add to message
    msg.data = compressed_depth
    # Set size of data
    msg.size = len(msg.data)
    # Set compression method
    msg.compression_method = image_t.COMPRESSION_METHOD_ZLIB
    return msg

def pack_rgbd(image: np.ndarray, depth: np.ndarray, name: str = "") -> rgbd_t:
    """!
    Packs an RGB image and a depth map into a single rgbd_t message.

    This function uses `pack_image` to encode the RGB image and `pack_depth` to encode the depth image,
    combining them into a single RGB-D message structure.

    @param image The RGB image (typically a NumPy array) to be packed.
    @param depth The depth image (typically a NumPy array) to be packed.
    @param name An optional name used to generate frame names for both components.
    @return An rgbd_t message containing the packed RGB and depth data.
    """
    image_msg = pack_image(image, name=name)
    depth_msg = pack_depth(depth, name=name)

    msg = rgbd_t()
    msg.image = image_msg
    msg.depth = depth_msg
    return msg 

def pack_laser_scan(angles: np.ndarray, ranges: np.ndarray) -> laser_scan_t:
    """!
    Pack angle and range data into a laser_scan_t message.

    This function uses pack_float_vector to wrap both arrays into the appropriate message format.

    @param angles  A 1D NumPy array of LiDAR angles.
    @param ranges  A 1D NumPy array of LiDAR ranges corresponding to the angles.
    @return A laser_scan_t message containing the packed angle and range data.
    """
    angles_msg = pack_float_vector(angles)
    ranges_msg = pack_float_vector(ranges)

    msg = laser_scan_t()
    msg.angles = angles_msg
    msg.ranges = ranges_msg 
    return msg

def pack_pose_2d(x: float, y: float, theta: float) -> pose_2d_t:
    """!
    Pack 2D pose information into a pose_2d_t message.

    @param x       The x-coordinate of the pose.
    @param y       The y-coordinate of the pose.
    @param theta   The orientation angle (in radians) of the pose.
    @return A pose_2d_t message with the x, y, and theta fields populated.
    """
    msg = pose_2d_t()
    msg.x = x
    msg.y = y
    msg.theta = theta
    return msg

def pack_velocity_2d(linear_velocity: float, angular_velocity: float) -> velocity_2d_t:
    """!
    Pack 2D velocity information into a velocity_t message.

    @param linear_velocity: The linear velocity component.
    @param angular_velocity: The angular velocity component.
    @return A velocity_t message with linear and angular velocity fields populated.
    """
    msg = velocity_2d_t()
    msg.linear = linear_velocity
    msg.angular = angular_velocity
    return msg

def pack_joint_group_command(cmd: List, name: str) -> joint_group_command_t:
    """!
    Pack joint group command into a joint_ground_command_t message.

    @param cmd     A list of the joint commands
    @param name    Name of the joint group.
    @return A joint_ground_command_t message
    """
    msg = joint_group_command_t()
    msg.name = name
    msg.n = len(cmd)
    msg.cmd = cmd
    return msg

def pack_grid_config(scene_bounds: Dict[str, List[float]], grid_size: float) -> grid_config_t:
    """!
    Pack grid_config into a grid_config_t message.

    @param scene_bounds A dictionary with 'x' and 'y' keys containing the min, max bounds for the scene.
    @param grid_size    The size of each grid cell.
    @return A grid_config_t message populated with the given bounds and grid size.
    """
    msg = grid_config_t()
    msg.x_bounds = scene_bounds["x"]
    msg.y_bounds = scene_bounds["y"]
    msg.grid_size = grid_size
    return msg

def pack_wheel_config(radius: float, thread: float) -> wheel_config_t:
    """!
    Packs wheel configuration data into a wheel_config_t message.

    Constructs a wheel_config_t message using provided wheel metadata:
    @param radius      Radius of the wheels.
    @param thread      Distance between the wheels 
    @return A wheel_config_t message containing the packed configuration data.
    """
    msg = wheel_config_t()
    msg.radius = radius
    msg.thread = thread
    return msg

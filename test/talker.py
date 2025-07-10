import numpy as np
from math import pi
from lcm import LCM
from time import sleep
from arktypes import transform_t
from scipy.spatial.transform import Rotation as Rot


def generate_transform(t):
    # Create translation
    x = np.sin(2 * pi * t)
    y = 0.5 * np.cos(2 * pi * t)
    z = -0.5 * np.sin(2 * pi * t)
    tr = [x, y, z]

    # Create rotation
    rz = pi * np.sin(2 * pi * t)
    ro = Rot.from_euler("xyz", [0, 0, rz])

    return transform_t.from_arrays(tr, ro)


def main():
    # Setup
    t = 0.0
    dt = 0.1
    lcm = LCM()

    # Start main loop
    try:
        while True:
            tf = generate_transform(t)
            lcm.publish("TEST_TRANSFORM", tf.encode())
            print(f"Published transform at t={t:.2f}")
            t += dt
            sleep(dt)
    except KeyboardInterrupt:
        print("complete")


if __name__ == "__main__":
    main()

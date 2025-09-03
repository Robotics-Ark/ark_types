from lcm import LCM
import numpy as np
from arktypes import transform_t

np.set_printoptions(precision=3, suppress=True, linewidth=1000)


def callback(channel, data):
    tf = transform_t.decode(data).as_array()
    print("======")
    print(f"Received transform on channel '{channel}'\n{tf}")


def main():
    lcm = LCM()
    lcm.subscribe("TEST_TRANSFORM", callback)

    print("Listening for messages on channel 'TEST_TRANSFORM'...")
    try:
        while True:
            lcm.handle()
    except KeyboardInterrupt:
        print("complete")


if __name__ == "__main__":
    main()

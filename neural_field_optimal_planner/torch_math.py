import numpy as np


def wrap_angle(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi

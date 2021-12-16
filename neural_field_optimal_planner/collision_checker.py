import numpy as np


class CollisionChecker(object):
    def __init__(self, robot_radius, collision_boundaries=None):
        self._obstacle_points = np.zeros((0, 3))
        self._boundaries = collision_boundaries
        self._robot_radius = robot_radius

    def check_collision(self, test_positions):
        distances = np.linalg.norm(test_positions[None] - self._obstacle_points[:, None], axis=2)
        result = np.any(distances < self._robot_radius, axis=0)
        if self._boundaries is not None:
            result |= test_positions[:, 0] > self._boundaries[1]
            result |= test_positions[:, 0] < self._boundaries[0]
            result |= test_positions[:, 1] > self._boundaries[3]
            result |= test_positions[:, 1] < self._boundaries[2]
        return result

    def update_obstacle_points(self, points):
        self._obstacle_points = points

    def update_boundaries(self, boundaries):
        self._boundaries = boundaries

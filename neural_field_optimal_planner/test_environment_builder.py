import numpy as np

from .test_environment import TestEnvironment


class TestEnvironmentBuilder(object):
    @staticmethod
    def make_test_environment():
        goal_point = np.array([2.5, 2.5], dtype=np.float32)
        start_point = np.array([0.5, 0.5], dtype=np.float32)
        trajectory_boundaries = (-0.1, 3.1, -0.1, 3.1)
        return TestEnvironment(start_point, goal_point, trajectory_boundaries,
                               TestEnvironmentBuilder._obstacle_points())

    @staticmethod
    def _obstacle_points():
        collision_points1_y = np.linspace(0, 2, 10)
        collision_points1 = np.stack([np.ones(10) * 1.15, collision_points1_y], axis=1)
        collision_points2 = collision_points1.copy()
        collision_points2[:, 0] = 1.85
        collision_points2[:, 1] += 1
        return np.concatenate([collision_points1, collision_points2], axis=0)

    @staticmethod
    def _point_line(start, end, point_count):
        x = np.linspace(start[0], end[0], point_count)
        y = np.linspace(start[1], end[1], point_count)
        return np.stack([x, y], axis=1)

    @staticmethod
    def make_test_environment_with_angles():
        goal_point = np.array([2.5, 1.5, 0], dtype=np.float32)
        start_point = np.array([0.5, 0.5, 0], dtype=np.float32)
        trajectory_boundaries = (-0.1, 3.1, -0.1, 3.1)
        return TestEnvironment(start_point, goal_point, trajectory_boundaries,
                               TestEnvironmentBuilder._obstacle_points())

    def make_car_environment(self):
        goal_point = np.array([2, 2.7, 0], dtype=np.float32)
        start_point = np.array([0.5, 1.5, 0], dtype=np.float32)
        trajectory_boundaries = (-0.1, 3.1, -0.1, 3.1)
        return TestEnvironment(start_point, goal_point, trajectory_boundaries,
                               TestEnvironmentBuilder._car_obstacle_points())

    @staticmethod
    def _car_obstacle_points():
        y1 = 2.3
        x1 = 1.6
        return np.concatenate([
            TestEnvironmentBuilder._point_line((0, y1), (x1, y1), 10),
            TestEnvironmentBuilder._point_line((x1, y1), (x1, 3), 10),
            TestEnvironmentBuilder._point_line((2.5, y1), (2.5, 3), 10),
            TestEnvironmentBuilder._point_line((2.5, y1), (3, y1), 10),
        ])
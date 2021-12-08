import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt

from neural_field_optimal_planner.collision_checker import CollisionChecker
from neural_field_optimal_planner.nerf_opt_planner import NERFOptPlanner
from neural_field_optimal_planner.onf_model import ONF
from neural_field_optimal_planner.plotting_utils import plot_planner_data, prepare_figure


class TestNERFOptPlanner(unittest.TestCase):
    def setUp(self) -> None:
        collision_model = ONF(1.5, 1)
        collision_checker = CollisionChecker(0.3, (0, 3, 0, 3))
        collision_checker.update_obstacle_points(self._obstacle_points())
        collision_optimizer = torch.optim.Adam(collision_model.parameters(), 1e-2)
        trajectory = torch.zeros(100, 2, requires_grad=True)
        trajectory_optimizer = torch.optim.Adam([trajectory], 1e-2)
        self._planner = NERFOptPlanner(trajectory, collision_model, collision_checker, collision_optimizer,
                                       trajectory_optimizer, 0.02, 0.5, 1)
        self._goal_point = np.array([2.5, 2.5], dtype=np.float32)
        self._start_point = np.array([0.5, 0.5], dtype=np.float32)
        self._trajectory_boundaries = (-0.1, 3.1, -0.1, 3.1)

    @staticmethod
    def _obstacle_points():
        collision_points1_y = np.linspace(0, 2, 10)
        collision_points1 = np.stack([np.ones(10) * 1.15, collision_points1_y], axis=1)
        collision_points2 = collision_points1.copy()
        collision_points2[:, 0] = 1.85
        collision_points2[:, 1] += 1
        return np.concatenate([collision_points1, collision_points2], axis=0)

    def test_init(self):
        self._planner.init(self._start_point, self._goal_point, self._trajectory_boundaries)
        self.assertNotEqual(self._planner.get_path()[0, 0], self._planner._start_point[0, 0])
        self.assertNotEqual(self._planner.get_path()[0, 1], self._planner._start_point[0, 1])
        self.assertNotEqual(self._planner.get_path()[-1, 0], self._planner._goal_point[0, 0])
        self.assertNotEqual(self._planner.get_path()[-1, 1], self._planner._goal_point[0, 1])

    def test_get_path(self):
        path = self._planner.get_path()
        self.assertEqual(path.shape, (100, 2))

    def test_full_path(self):
        full_path = self._planner.full_trajectory()
        self.assertEqual(full_path.shape, (102, 2))

    def test_step(self):
        self._planner.init(self._start_point, self._goal_point, self._trajectory_boundaries)
        self._planner.step()

    def test_100step(self):
        self._planner.init(self._start_point, self._goal_point, self._trajectory_boundaries)
        for i in range(100):
            self._planner.step()

    def test_step_with_visualization(self):
        self._planner.init(self._start_point, self._goal_point, self._trajectory_boundaries)
        fig = None
        for i in range(10):
            self._planner.step()
            trajectory = self._planner.get_path()
            collision_model = self._planner._collision_model
            obstacle_points = self._planner._collision_checker._obstacle_points
            boundaries = self._trajectory_boundaries

            if fig is not None:
                fig.clear()
            fig = prepare_figure(boundaries, 1)
            plot_planner_data(trajectory, collision_model, boundaries, obstacle_points)
            plt.pause(0.01)

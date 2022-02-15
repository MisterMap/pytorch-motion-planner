import torch

from .torch_math import wrap_angle


class TrajectoryInitializer(object):
    def __init__(self, collision_checker):
        self._collision_checker = collision_checker

    def initialize_trajectory(self, trajectory, start_point, goal_point):
        with torch.no_grad():
            trajectory_length = trajectory.shape[0] + 2
            trajectory[:, 0] = torch.linspace(
                start_point[0, 0], goal_point[0, 0], trajectory_length)[1:-1]
            trajectory[:, 1] = torch.linspace(
                start_point[0, 1], goal_point[0, 1], trajectory_length)[1:-1]
            self.initialize_angle(trajectory, start_point, goal_point)

    @staticmethod
    def initialize_angle(trajectory, start_point, goal_point):
        with torch.no_grad():
            trajectory_length = trajectory.shape[0] + 2
            delta_angle = wrap_angle(goal_point[0, 2] - start_point[0, 2])
            goal_angle = delta_angle + start_point[0, 2]
            trajectory[:, 2] = torch.linspace(
                start_point[0, 2], goal_angle, trajectory_length)[1:-1]

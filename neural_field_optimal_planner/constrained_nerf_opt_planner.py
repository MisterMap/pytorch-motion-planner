import numpy as np
import torch

from .nerf_opt_planner import NERFOptPlanner
from .torch_math import wrap_angle


class ConstrainedNERFOptPlanner(NERFOptPlanner):
    def __init__(self, trajectory, collision_model, collision_checker, collision_optimizer, trajectory_optimizer,
                 trajectory_random_offset, collision_weight, velocity_hessian_weight, init_collision_iteration=100,
                 init_collision_points=100, reparametrize_trajectory_freq=10, optimize_collision_model_freq=1,
                 random_field_points=10, angle_weight=0.5, constraint_deltas_weight=20, multipliers_lr=1e-1):
        super().__init__(trajectory, collision_model, collision_checker, collision_optimizer, trajectory_optimizer,
                         trajectory_random_offset, collision_weight, velocity_hessian_weight, init_collision_iteration,
                         init_collision_points, reparametrize_trajectory_freq, optimize_collision_model_freq,
                         random_field_points)
        self._goal_point = torch.zeros(1, 3, device=self._device)
        self._start_point = torch.zeros(1, 3, device=self._device)
        self._angle_weight = angle_weight
        self._constraint_multipliers = torch.zeros(self._trajectory.shape[0] + 1, device=self._device,
                                                   requires_grad=True)
        self._constraint_multipliers.grad = None
        self._constraint_delta_weight = constraint_deltas_weight
        self._multipliers_lr = multipliers_lr

    def _random_intermediate_positions(self):
        t = torch.tensor(np.random.rand(self._trajectory.shape[0] - 1).astype(np.float32), device=self._device)[:, None]
        return self._trajectory[1:, :2] * (1 - t) + self._trajectory[:-1, :2] * t

    def _init_trajectory(self):
        super()._init_trajectory()
        with torch.no_grad():
            trajectory_length = self._trajectory.shape[0] + 2
            self._trajectory[:, 2] = torch.linspace(
                self._start_point[0, 2], self._goal_point[0, 2], trajectory_length)[1:-1]

    def _calculate_distances(self):
        full_trajectory = self.full_trajectory()
        return torch.norm(full_trajectory[1:, :2] - full_trajectory[:-1, :2], dim=1)

    def non_holonomic_constraint_deltas(self):
        full_trajectory = self.full_trajectory()
        dx = full_trajectory[1:, 0] - full_trajectory[:-1, 0]
        dy = full_trajectory[1:, 1] - full_trajectory[:-1, 1]
        angles = full_trajectory[:, 2]
        delta_angles = angles[1:] - angles[:-1]
        mean_angles = angles[:-1] + delta_angles / 2
        return dx * torch.sin(mean_angles) - dy * torch.cos(mean_angles)

    def direction_constraint_deltas(self):
        full_trajectory = self.full_trajectory()
        dx = full_trajectory[1:, 0] - full_trajectory[:-1, 0]
        dy = full_trajectory[1:, 1] - full_trajectory[:-1, 1]
        angles = full_trajectory[:, 2]
        delta_angles = wrap_angle(angles[:-1] - angles[1:])
        mean_angles = angles[:-1] + delta_angles / 2
        return -(torch.cos(mean_angles) * dx + torch.sin(mean_angles) * dy)

    def distance_loss(self):
        full_trajectory = self.full_trajectory()
        delta = full_trajectory[1:] - full_trajectory[:-1]
        delta_angles = delta[:, 2]
        delta_angles = wrap_angle(delta_angles)
        angle_sum = torch.sum(delta_angles.detach()) - full_trajectory[-1, 2] + full_trajectory[0, 2]
        delta[-1, 2] += angle_sum
        delta[:, 2] *= self._angle_weight
        return torch.sum(delta ** 2)

    def trajectory_loss(self):
        collision_positions = self._random_intermediate_positions()
        constraint_deltas = self.non_holonomic_constraint_deltas()
        loss = self.distance_loss() + self.trajectory_collision_loss(collision_positions) * self._collision_weight + \
               torch.sum(self._constraint_multipliers * constraint_deltas) + torch.sum(constraint_deltas ** 2) * \
               self._constraint_delta_weight
        return loss

    def _optimize_trajectory(self):
        super()._optimize_trajectory()
        self._constraint_multipliers.data.add_(self._multipliers_lr * self._constraint_multipliers.grad.detach())
        self._constraint_multipliers.grad = None

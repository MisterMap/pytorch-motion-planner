import numpy as np
import torch
import torch.nn.functional
from torch import nn

from .nerf_opt_planner import NERFOptPlanner
from .torch_math import wrap_angle


class ConstrainedNERFOptPlanner(NERFOptPlanner):
    def __init__(self, trajectory, collision_model, collision_checker, collision_optimizer, trajectory_optimizer,
                 trajectory_random_offset, collision_weight, velocity_hessian_weight, init_collision_iteration=100,
                 init_collision_points=100, reparametrize_trajectory_freq=10, optimize_collision_model_freq=1,
                 random_field_points=10, angle_weight=0.5, constraint_deltas_weight=20, multipliers_lr=1e-1,
                 boundary_weight=1, collision_multipliers_lr=1e-3):
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
        self._collision_multipliers_lr = collision_multipliers_lr
        self._boundary_weight = boundary_weight
        self._collision_multipliers = torch.zeros(self._trajectory.shape[0], device=self._device,
                                                  requires_grad=True)
        self._collision_multipliers.grad = None

    def _random_intermediate_positions(self, trajectory=None):
        if trajectory is None:
            trajectory = self._trajectory
        t = torch.tensor(np.random.rand(trajectory.shape[0] - 1).astype(np.float32), device=self._device)[:, None]
        return trajectory[1:, :2] * (1 - t) + trajectory[:-1, :2] * t

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
        t = torch.tensor(np.random.rand(self._trajectory.shape[0] - 1).astype(np.float32), device=self._device)[:, None]
        collision_positions = self._trajectory[1:, :2] * (1 - t) + self._trajectory[:-1, :2] * t
        collision_multipliers = self._collision_multipliers[1:] * (1 - t[:, 0]) + self._collision_multipliers[:-1] * t[
        :, 0]
        collision_probabilities = self._collision_model(collision_positions)
        softplus_collision_probabilities = nn.functional.softplus(collision_probabilities)
        collision_multipliers_loss = torch.sum(collision_multipliers * torch.tanh(collision_probabilities[:, 0]))
        collision_loss = torch.sum(softplus_collision_probabilities)

        constraint_deltas = self.non_holonomic_constraint_deltas()
        loss = self.distance_loss() + collision_loss * self._collision_weight + \
               torch.sum(self._constraint_multipliers * constraint_deltas) + torch.sum(constraint_deltas ** 2) * \
               self._constraint_delta_weight + self.boundary_loss() * self._boundary_weight + collision_multipliers_loss
        return loss

    def trajectory_collision_loss(self, collision_positions):
        collision_probabilities = self._collision_model(collision_positions)
        collision_probabilities = nn.functional.softplus(collision_probabilities)
        return torch.sum(collision_probabilities)

    def _optimize_trajectory(self):
        super()._optimize_trajectory()
        self._constraint_multipliers.data.add_(self._multipliers_lr * self._constraint_multipliers.grad.detach())
        self._constraint_multipliers.grad = None
        self._collision_multipliers.data.add_(
            self._collision_multipliers_lr * self._collision_multipliers.grad.detach())
        with torch.no_grad():
            self._collision_multipliers.data = torch.where(self._collision_multipliers > 0, self._collision_multipliers,
                                                           torch.zeros_like(self._collision_multipliers))
        self._collision_multipliers.grad = None

    def reparametrize_trajectory(self):
        full_trajectory = self.full_trajectory()
        distances = self._calculate_distances()
        normalized_distances = distances / torch.sum(distances)
        cdf = torch.cumsum(normalized_distances, dim=0)
        cdf = torch.cat([torch.zeros(1, device=self._device), cdf], dim=0)
        uniform_samples = torch.linspace(0, 1, len(full_trajectory), device=self._device)[1:-1]
        indices = torch.searchsorted(cdf, uniform_samples)
        index_above = torch.where(indices > len(full_trajectory) - 1, len(full_trajectory) - 1, indices)
        index_bellow = torch.where(indices - 1 < 0, 0, indices - 1)
        cdf_above = torch.gather(cdf, 0, index_above)
        cdf_bellow = torch.gather(cdf, 0, index_bellow)
        index_above = torch.repeat_interleave(index_above[:, None], self._trajectory.shape[1], dim=1)
        index_bellow = torch.repeat_interleave(index_bellow[:, None], self._trajectory.shape[1], dim=1)
        trajectory_above = torch.gather(full_trajectory, 0, index_above)
        trajectory_bellow = torch.gather(full_trajectory, 0, index_bellow)
        denominator = cdf_above - cdf_bellow
        denominator = torch.where(denominator < 1e-5, torch.ones_like(denominator) * 1e-5, denominator)
        t = (uniform_samples - cdf_bellow) / denominator
        trajectory = (1 - t[:, None]) * trajectory_bellow + t[:, None] * trajectory_above
        self._trajectory.data = trajectory
        collision_multipliers = torch.cat([torch.zeros(1, device=self._device), self._collision_multipliers,
                                              torch.zeros(1, device=self._device)])
        collision_multipliers_above = torch.gather(collision_multipliers, 0, index_above[:, 0])
        collision_multipliers_bellow = torch.gather(collision_multipliers, 0, index_bellow[:, 0])
        self._collision_multipliers.data = (1 - t) * collision_multipliers_bellow + t * collision_multipliers_above

        constraint_multipliers = torch.cat(
            [torch.full((1,), self._constraint_multipliers[0].item(), device=self._device),
                (self._constraint_multipliers[:-1] + self._constraint_multipliers[1:]) / 2,
                torch.full((1,), self._constraint_multipliers[-1].item(), device=self._device)])
        constraint_multiplier_above = torch.gather(constraint_multipliers, 0, index_above[:, 0])
        constraint_multiplier_bellow = torch.gather(constraint_multipliers, 0, index_bellow[:, 0])
        constraint_multipliers = (1 - t) * constraint_multiplier_bellow + t * constraint_multiplier_above
        self._constraint_multipliers.data = torch.cat(
            [torch.full((1,), constraint_multipliers[0].item(), device=self._device),
                (constraint_multipliers[:-1] + constraint_multipliers[1:]) / 2,
                torch.full((1,), constraint_multipliers[-1].item(), device=self._device)])

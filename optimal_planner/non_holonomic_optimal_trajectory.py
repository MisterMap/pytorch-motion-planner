import numpy as np

from .optimal_trajectory import OptimalTrajectory


class NonHolonomicOptimalTrajectory(OptimalTrajectory):
    def __init__(self, trajectory, angle_weight=1, constraint_weight=1000):
        super().__init__(trajectory, angle_weight)
        self._constraint_weight = constraint_weight

    def loss(self):
        return self.distance_loss() + self.constraint_loss() * self._constraint_weight

    def constraint_loss(self):
        constraint_deltas = self.constraint_deltas()
        return np.sum(constraint_deltas ** 2)

    def constraint_deltas(self):
        dx = self.x[1:] - self.x[:-1]
        dy = self.y[1:] - self.y[:-1]
        return dx * np.sin(self.angle[:-1]) - dy * np.cos(self.angle[:-1])

    def jacobian(self):
        return self.distance_jacobian() + self.constraint_jacobian() * self._constraint_weight

    def constraint_jacobian(self):
        constraint_deltas = self.constraint_deltas()
        x_jacobian = np.sin(self.angle[:-1]) * constraint_deltas
        x_jacobian = x_jacobian[:-1] - x_jacobian[1:]
        y_jacobian = -np.cos(self.angle[:-1]) * constraint_deltas
        y_jacobian = y_jacobian[:-1] - y_jacobian[1:]
        dx = self.x[2:] - self.x[1:-1]
        dy = self.y[2:] - self.y[1:-1]
        angle_jacobian = dx * np.cos(self.angle[1:-1]) + dy * np.sin(self.angle[1:-1])
        angle_jacobian = angle_jacobian * constraint_deltas[1:]
        return 2 * np.stack([x_jacobian, y_jacobian, angle_jacobian], axis=1).reshape(-1)


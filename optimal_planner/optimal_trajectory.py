import numpy as np

from .math import wrap_angle


class OptimalTrajectory(object):
    def __init__(self, trajectory, angle_koef):
        self._point_count = len(trajectory)
        self._angle_koef = angle_koef
        self._trajectory = trajectory

    def loss(self):
        return self.distance_loss()

    def distance_loss(self):
        x_loss = self._mse_delta_loss(self.x)
        y_loss = self._mse_delta_loss(self.y)
        angle_loss = self._mse_delta_loss(self.angle, need_wrap=True)
        return x_loss + y_loss + angle_loss * self._angle_koef

    def _mse_delta_loss(self, x, need_wrap=False):
        delta = self._delta_array(x, need_wrap)
        return np.sum(delta ** 2)

    @staticmethod
    def _delta_array(x, need_wrap=False):
        delta = np.roll(x, -1, axis=0) - x
        if need_wrap:
            delta = wrap_angle(delta)
        return delta[:-1]

    @property
    def x(self):
        return self._trajectory[:, 0]

    @property
    def y(self):
        return self._trajectory[:, 1]

    @property
    def angle(self):
        return self._trajectory[:, 2]

    def parameters(self):
        return self._trajectory[1:-1].reshape(-1)

    @classmethod
    def from_parameters(cls, parameters, start_point, goal_point, angle_koef=1.):
        trajectory = np.concatenate([start_point[None], parameters.reshape(-1, 3), goal_point[None]], axis=0)
        return cls(trajectory, angle_koef)

    def jacobian(self):
        return self.distance_jacobian()

    def distance_jacobian(self):
        x_jacobian = self._mse_delta_jacobian(self.x)
        y_jacobian = self._mse_delta_jacobian(self.y)
        angle_jacobian = self._mse_delta_jacobian(self.angle, need_wrap=True) * self._angle_koef
        return np.stack([x_jacobian, y_jacobian, angle_jacobian], axis=1).reshape(-1)

    def _mse_delta_jacobian(self, x, need_wrap=False):
        delta = self._delta_array(x, need_wrap)
        return 2 * (delta[:-1] - delta[1:])

    def hessian(self):
        return self.distance_hessian()

    def distance_hessian(self):
        result = np.zeros((self._point_count - 2, 3, self._point_count - 2, 3))
        result[:, 0, :, 0] = self._mse_delta_hessian(self.x[1:-1])
        result[:, 1, :, 1] = self._mse_delta_hessian(self.y[1:-1])
        result[:, 2, :, 2] = self._mse_delta_hessian(self.angle[1:-1]) * self._angle_koef
        return result.reshape(3 * self._point_count - 6, 3 * self._point_count - 6)

    @staticmethod
    def _mse_delta_hessian(x):
        result = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            result[i, i] = 4
            if i > 0:
                result[i, i-1] = -2
                result[i - 1, i] = -2
        return result

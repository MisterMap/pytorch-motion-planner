from .utils.math import cvt_global2local, cvt_local2global, wrap_angle
import numpy as np


class PointsParametrizationEnergyStateSpace(object):
    def __init__(self, points, start_delta_angles, cost_radius):
        self._points = points
        self._start_delta_angle = start_delta_angles
        self._cost_radius = cost_radius

    @property
    def cost_radius(self):
        return self._cost_radius

    def set_parametrization(self, parameters):
        self._points = np.array(parameters[1:]).reshape(-1, 2)
        self._start_delta_angle = parameters[0]

    def trajectory(self, start_position, goal_position):
        interpolated_points = np.zeros((0, 3))
        current_point = start_position.copy().astype(np.float)
        current_point[2] += self._start_delta_angle
        points = np.concatenate((self._points, np.array([goal_position[:2]])), axis=0)
        for point in points:
            r, dangle = self.get_circle_parameters(current_point, point)
            angles = np.linspace(0, dangle, 10)
            new_points = np.array([r * np.sin(angles), r * (1 - np.cos(angles)), angles]).T
            new_points = cvt_local2global(new_points, current_point)
            new_angle = current_point[2] + dangle
            current_point[:2] = point
            current_point[2] = new_angle
            interpolated_points = np.concatenate((interpolated_points, new_points), axis=0)
        return interpolated_points

    def distance(self, start_position, goal_position):
        points = np.concatenate((self._points, np.array([goal_position[:2]])), axis=0)
        current_point = start_position.copy().astype(np.float)
        current_point[2] += self._start_delta_angle
        cost = np.abs(wrap_angle(self._start_delta_angle)) * self._cost_radius
        for point in points:
            r, dangle = self.get_circle_parameters(current_point, point)
            cost += np.abs(dangle) * max(np.abs(r), self._cost_radius)
            new_angle = current_point[2] + dangle
            current_point[:2] = point
            current_point[2] = new_angle
        cost += np.abs(wrap_angle(current_point[2] - goal_position[2])) * self._cost_radius
        return cost

    @staticmethod
    def get_circle_parameters(start_position, goal_position, eps=1e-6):
        delta_point = cvt_global2local(goal_position, start_position)
        dx = delta_point[0]
        dy = delta_point[1]
        r = np.sign(dy) * (dy ** 2 + dx ** 2) / (2 * np.abs(dy) + eps)
        delta_angle = np.arctan2(2 * dx * dy / (dx ** 2 + dy ** 2 + eps), 1 - 2 * dy ** 2 / (dx ** 2 + dy ** 2 + eps))
        if dx <= 0:
            delta_angle += np.sign(dy) * 2 * np.pi
        return r, delta_angle

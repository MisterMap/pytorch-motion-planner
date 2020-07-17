import numpy as np
from .utils.math import cvt_local2global, cvt_global2local, wrap_angle


class EnergyConstCurvaturesStateSpace(object):
    def __init__(self, radiuses, delta_alpha, delta_length, cost_radius):
        self._radiuses = radiuses
        self._delta_alpha = delta_alpha
        self._delta_length = delta_length
        self._cost_radius = cost_radius
        self._points = self._make_points()

    def set_radiuses(self, radiuses):
        self._radiuses = radiuses
        self._points = self._make_points()

    def _make_points(self):
        points = [np.array([0, 0, 0])]
        for radius in self._radiuses:
            dangle = np.sign(radius) * min(self._delta_alpha, self._delta_length / np.abs(radius))
            dx = radius * np.sin(dangle)
            dy = radius * (1 - np.cos(dangle))
            new_point = cvt_local2global(np.array([dx, dy, dangle]), points[-1])
            points.append(new_point)
        return np.array(points)

    def _transform_parameters(self, start_point, finish_point):
        target_point = cvt_global2local(finish_point, start_point)
        scale_koef = np.linalg.norm(target_point[:2]) / np.linalg.norm(self._points[-1, :2])
        delta_angle = np.arctan2(target_point[1], target_point[0]) - np.arctan2(
            self._points[-1, 1], self._points[-1, 0])
        return scale_koef, delta_angle

    def trajectory(self, start_point, finish_point):
        scale_koef, delta_angle = self._transform_parameters(start_point, finish_point)
        new_xs = self._points[:, 0] * scale_koef
        new_ys = self._points[:, 1] * scale_koef
        transformed_points = np.array([new_xs, new_ys, self._points[:, 2]]).T
        transformed_points = cvt_local2global(transformed_points, np.array([0, 0, delta_angle]))
        transformed_points = cvt_local2global(transformed_points, start_point)
        return transformed_points

    def distance(self, start_point, finish_point):
        target_point = cvt_global2local(finish_point, start_point)
        scale_koef, delta_angle = self._transform_parameters(start_point, finish_point)
        dangle = np.minimum(self._delta_alpha, self._delta_length / np.abs(self._radiuses))
        cost = np.sum(np.maximum(np.abs(self._radiuses * scale_koef), self._cost_radius) * dangle)
        cost += np.abs(wrap_angle(delta_angle)) * self._cost_radius
        cost += np.abs(wrap_angle(self._points[-1, 2] + delta_angle - target_point[2])) * self._cost_radius
        return cost

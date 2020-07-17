import numpy as np
from .utils.math import cvt_local2global, cvt_global2local, wrap_angle


class AngleCurvatureParametrizationStateSpace(object):
    def __init__(self, radiuses, angles, cost_radius):
        self._radiuses = radiuses
        self._angles = np.sign(radiuses) * np.abs(angles)
        self._cost_radius = cost_radius
        self._points = self._make_points()

    def set_parametrization(self, parametrization):
        point_count = parametrization.shape[0] // 2
        self._radiuses = parametrization[:point_count]
        self._angles = np.sign(self._radiuses) * np.abs(parametrization[point_count:])
        self._points = self._make_points()

    def _make_points(self):
        points = np.zeros((1, 3))
        for radius, angle in zip(self._radiuses, self._angles):
            angles = np.linspace(0, angle, 10)
            dx = radius * np.sin(angles)
            dy = radius * (1 - np.cos(angles))
            new_points = cvt_local2global(np.array([dx, dy, angles]).T, points[-1])
            points = np.concatenate((points, new_points), axis=0)
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
        dangle = np.abs(self._angles)
        cost = np.sum(np.maximum(np.abs(self._radiuses * scale_koef), self._cost_radius) * dangle)
        cost += np.abs(wrap_angle(delta_angle)) * self._cost_radius
        cost += np.abs(wrap_angle(self._points[-1, 2] + delta_angle - target_point[2])) * self._cost_radius
        return cost

    # def get_line_cost(self, start_point, finish_point):
    #     delta = finish_point[:2] - start_point[:2]
    #     direction_angle = np.arctan2(delta[1], delta[0])
    #     cost = np.linalg.norm(delta)
    #     cost += self._cost_radius * np.abs(wrap_angle(start_point[2] - direction_angle))
    #     cost += self._cost_radius * np.abs(wrap_angle(finish_point[2] - direction_angle))
    #     return cost

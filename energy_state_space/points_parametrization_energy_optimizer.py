import numpy as np
import torch

from .utils.math import torch_global2local, wrap_angle


class PointsParametrizationEnergyOptimizer(object):
    def __init__(self, state_space):
        self._state_space = state_space

    def cost_function(self, points, start_delta_angle, start_point, goal_point):
        points = torch.cat((points, goal_point[:2][None]), dim=0)
        current_point = start_point.clone()
        current_point[2] += start_delta_angle
        cost = torch.abs(wrap_angle(start_delta_angle)) * self._state_space.cost_radius
        for point in points:
            r, dangle = self.get_circle_parameters(current_point, point)
            r = torch.abs(r) if torch.abs(r) > self._state_space.cost_radius else self._state_space.cost_radius
            cost += torch.abs(dangle) * r
            new_angle = current_point[2] + dangle
            current_point = torch.cat([point, new_angle[None]])
        cost += torch.abs(wrap_angle(current_point[2] - goal_point[2])) * self._state_space.cost_radius
        return cost

    @staticmethod
    def get_circle_parameters(start_point, finish_point, eps=1e-6):
        delta_point = torch_global2local(finish_point[None], start_point[None])[0]
        dx = delta_point[0]
        dy = delta_point[1]
        r = torch.sign(dy) * (dy ** 2 + dx ** 2) / (2 * torch.abs(dy) + eps)
        delta_angle = torch.atan2(2 * dx * dy / (dx ** 2 + dy ** 2 + eps), 1 - 2 * dy ** 2 / (dx ** 2 + dy ** 2 + eps))
        if dx <= 0:
            delta_angle += torch.sign(dy) * 2 * np.pi
        return r, delta_angle

    def optimize(self, start_point, goal_point, initial_points, initial_start_delta_angle, iteration_count):
        start_delta_angle = torch.tensor(initial_start_delta_angle, requires_grad=True)
        points = torch.tensor(initial_points, requires_grad=True)
        start_point = torch.tensor(start_point)
        goal_point = torch.tensor(goal_point)
        optimizer = torch.optim.SGD([points, start_delta_angle], lr=1e-3)
        for i in range(iteration_count):
            cost = self.cost_function(points, start_delta_angle, start_point, goal_point)
            cost.backward()
            with torch.no_grad():
                optimizer.step()
                points.grad.zero_()
                start_delta_angle.grad.zero_()
        self._state_space.set_parametrization([start_delta_angle.item()] + list(points.detach().numpy().reshape(-1)))

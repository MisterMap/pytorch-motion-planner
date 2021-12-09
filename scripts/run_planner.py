import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from neural_field_optimal_planner.collision_checker import CollisionChecker
from neural_field_optimal_planner.nerf_opt_planner import NERFOptPlanner
from neural_field_optimal_planner.onf_model import ONF
from neural_field_optimal_planner.plotting_utils import prepare_figure, plot_planner_data


def get_obstacle_points():
    collision_points1_y = np.linspace(0, 2, 10)
    collision_points1 = np.stack([np.ones(10) * 1.15, collision_points1_y], axis=1)
    collision_points2 = collision_points1.copy()
    collision_points2[:, 0] = 1.85
    collision_points2[:, 1] += 1
    return np.concatenate([collision_points1, collision_points2], axis=0)


torch.random.manual_seed(100)
np.random.seed(100)

device = "cpu"
collision_model = ONF(1.5, 1).to(device)
collision_checker = CollisionChecker(0.3, (0, 3, 0, 3))
obstacle_points = get_obstacle_points()
collision_checker.update_obstacle_points(obstacle_points)
collision_optimizer = torch.optim.Adam(collision_model.parameters(), 1e-2)
trajectory = torch.zeros(100, 2, requires_grad=True, device=device)
trajectory_optimizer = torch.optim.Adam([trajectory], 1e-2, betas=(0.9, 0.999))
planner = NERFOptPlanner(trajectory, collision_model, collision_checker, collision_optimizer,
                         trajectory_optimizer, trajectory_random_offset=0.02, collision_weight=0.01,
                         velocity_hessian_weight=3, random_field_points=10, init_collision_iteration=400)
goal_point = np.array([2.5, 2.5], dtype=np.float32)
start_point = np.array([0.5, 0.5], dtype=np.float32)
trajectory_boundaries = (-0.1, 3.1, -0.1, 3.1)

planner.init(start_point, goal_point, trajectory_boundaries)

fig = plt.figure(1, dpi=200)

for i in range(1000):
    planner.step()
    trajectory = planner.get_path()
    fig.clear()
    prepare_figure(trajectory_boundaries)
    plot_planner_data(trajectory, collision_model, trajectory_boundaries, obstacle_points, device=device)
    plt.pause(0.01)

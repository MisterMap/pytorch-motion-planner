import numpy as np
import torch
from matplotlib import pyplot as plt

from neural_field_optimal_planner.collision_checker import CollisionChecker
from neural_field_optimal_planner.planner_factory import PlannerFactory
from neural_field_optimal_planner.plotting_utils import prepare_figure, plot_planner_data, plot_nerf_opt_planner
from neural_field_optimal_planner.test_environment_builder import TestEnvironmentBuilder

torch.random.manual_seed(100)
np.random.seed(400)

test_environment = TestEnvironmentBuilder().make_test_environment_with_angles()
obstacle_points = test_environment.obstacle_points
collision_checker = CollisionChecker(0.3, (0, 3, 0, 3))
collision_checker.update_obstacle_points(test_environment.obstacle_points)

planner = PlannerFactory.make_constrained_onf_planner(collision_checker)
goal_point = test_environment.goal_point
start_point = test_environment.start_point
trajectory_boundaries = test_environment.bounds

planner.init(start_point, goal_point, trajectory_boundaries)
device = planner._device
collision_model = planner._collision_model
fig = plt.figure(1, dpi=200)

for i in range(1000):
    planner.step()
    trajectory = planner.get_path()
    fig.clear()
    prepare_figure(trajectory_boundaries)
    plot_planner_data(trajectory, collision_model, trajectory_boundaries, obstacle_points, device=device)
    plot_nerf_opt_planner(planner)
    plt.pause(0.01)

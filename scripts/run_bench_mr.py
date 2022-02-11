#! /usr/bin/python3.9
import numpy as np
import torch
from matplotlib import pyplot as plt
import argparse

from neural_field_optimal_planner.planner_factory import PlannerFactory
from neural_field_optimal_planner.plotting_utils import prepare_figure, plot_planner_data, plot_nerf_opt_planner
from neural_field_optimal_planner.benchmark_adapter import BenchmarkAdapter
from neural_field_optimal_planner.benchmark_adapter.benchmark_collision_checker import BenchmarkCollisionChecker

torch.random.manual_seed(100)
np.random.seed(400)

parser = argparse.ArgumentParser()
parser.add_argument("settings")
parser.add_argument("--show", default=False)
args = parser.parse_args()

print("start with config", args.settings)
benchmark = BenchmarkAdapter(args.settings)
collision_checker = BenchmarkCollisionChecker(benchmark, benchmark.bounds())

planner = PlannerFactory.make_constrained_onf_planner(collision_checker)
goal_point = benchmark.goal().as_vec()
start_point = benchmark.start().as_vec()
trajectory_boundaries = benchmark.bounds()

planner.init(start_point, goal_point, trajectory_boundaries)
device = planner._device
collision_model = planner._collision_model
is_show = args.show
fig = None
if is_show:
    fig = plt.figure(dpi=200)

for i in range(1000):
    planner.step()
    if is_show:
        trajectory = planner.get_path()
        fig.clear()
        prepare_figure(trajectory_boundaries)
        plot_planner_data(trajectory, collision_model, trajectory_boundaries, np.zeros((0, 2)), device=device)
        plot_nerf_opt_planner(planner)
        plt.pause(0.01)